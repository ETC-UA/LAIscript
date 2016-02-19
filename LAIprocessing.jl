using Logging
import StatsBase, JLD

const CAMERALENSES = "CameraLenses.jld"
if !isfile(CAMERALENSES)
    warn("file with previous CameraLens calibrations not found, will create empty one called $CAMERALENSES")
    close(JLD.jldopen(CAMERALENSES,"w"))
end

"sends variables to other processors"
function sendto(p::Int; args...)
    for (nm, val) in args
        @spawnat(p, eval(Main, Expr(:(=), nm, val)))
    end
end
function sendto(ps::Vector{Int}; args...)
    for p in ps
       sendto(p; args...)
    end
end

# Load required functions and packages on each processor
using LeafAreaIndex
using Images #also imports FileIO for reading jpg

@everywhere begin
    #Lg = Logging

    abstract LAIresultInfo
    "Convenience type to hold results from LAI calculation."
    type LAIresult <: LAIresultInfo
        imagepath::AbstractString
        LAI::Float64
        thresh::Float64
        clump::Float64
    end
    type NoLAIresult <: LAIresultInfo
        exception::Exception
    end

    function getLAI(imagepath::AbstractString, cl::LeafAreaIndex.CameraLens, 
                    sl::LeafAreaIndex.SlopeInfo)#, mainlogfile::AbstractString)        
        # This function gets executed in parallel, so need to set up new logger
        # on each processor.
        #@show "before logger"
        #baselog, logext = splitext(mainlogfile)
        #@show locallogfile = baselog * string(myid()) * logext
        #writecsv(locallogfile, "") #clear logfile
        #println("csv written")
        #@show locallog = Lg.Logger("locallog"*string(myid()))
        #Lg.configure(locallog, filename=locallogfile, level=DEBUG)
        try
            #@show i = myid() # ID of current processor            
            #Lg.debug(locallog, "start getLAI on $imagepath")            
            img = readrawjpg(imagepath, sl)            
            #@show "image read"
            #debug(setlog, "$i create PolarImage")
            polim = LeafAreaIndex.PolarImage(img, cl, sl)
            #debug(setlog, "$i PolarImage created")
            thresh= LeafAreaIndex.threshold(polim)
            #debug(setlog,"$i threshold: $thresh")        
            LAIe  = LeafAreaIndex.inverse(polim, thresh)
            #debug(setlog,"$i effective LAI: $LAIe")
            clump = LeafAreaIndex.langxiang45(polim, thresh, 0, pi/2)
            #debug(setlog,"$i clumping: $clump")
            LAI = LAIe / clump
            #debug(setlog,"$i LAI: $LAI")            
            return LAIresult(imagepath, LAI, thresh, clump)
        catch lai_err
            #debug(setlog,"$i error: $lai_err")
            #@show lai_err
            return NoLAIresult(lai_err)
        end
    end

    function readrawjpg(imp::AbstractString, sl::LeafAreaIndex.SlopeInfo)#, setlog::Logging.Logger)
        #i = myid()# ID of current processor
        #debug(setlog, "$i start reading $imp")
        @assert imp != nothing
        @assert isfile(imp)

        ext = lowercase(splitext(imp)[end])

        if ext in LeafAreaIndex.RAW_EXT
            img = LeafAreaIndex.rawblueread(imp)
            #@show img[1]
            # no need to keep Gray color or Images info
            imgblue = Images.data(reinterpret(FixedPointNumbers.UFixed16, img))
        elseif ext in [".jpg",".jpeg", ".tiff"]
            img = FileIO.load(imp)
            imgblue = Images.blue(img)
            gamma_decode!(imgblue)
        else
            warn("image has unknown extension at $imp")
            #warn(setlog,"$i image has unknown extension at $imp")
        end
        # images usually stored in horizontal mode, but LeafAreaIndex expects row-major Matrix
        imgblue = transpose(imgblue)
        #debug(setlog, "image read")

        #@show "check overexposure"
        if sum(imgblue .== 1) > 0.005 * length(imgblue)
            #warn("Image overexposed: $imp")
            #warn(setlog, "$i Image overexposed: $imp")
        end

        #rotate if in portrait mode
        if size(imgblue,1) > size(imgblue,2)
            if LeafAreaIndex.slope(sl) != zero(LeafAreaIndex.slope(sl))                
                #warn(setlog, "$i image with slope in portrait mode, don't know which way to turn: $imp")
                error("image with slope in portrait mode, don't know which way to turn: $imp")
            end
            imgblue = rotate90(imgblue) #default clockwise, could influence result due to lens center
        end
        #println("return rrj")
        return imgblue
    end

    "Rotates an image (or in general an `AbstractMatrix`) 90 degrees."
    # Rotate sometimes because currently LeafAreaIndex expects landscape *in memory*.
    function rotate90(img::AbstractMatrix; clockwise=true)
        img = transpose(img)
        if clockwise
            # flip up-down
            for i = 1:size(img, 2)
                img[:, i] = reverse(img[:, i])
            end
        else #counterclockwise
            # flip left-right
            flip = similar(img)
            for i = 1:size(img, 2)
                flipcol = size(img,2) + 1 - i
                flip[:, flipcol] = img[:, 1]
            end
            img = flip
        end
        img
    end

    "Gamma decode a gray image taken from single channel in sRGB colorspace."
    function gamma_decode!(A::AbstractMatrix)
    @fastmath for j = 1:size(A, 2)
        for i = 1:size(A, 1)
            # See https://en.wikipedia.org/wiki/SRGB
            A[i,j] = A[i,j] <= 0.04045 ? A[i,j]/12.92 : ((A[i,j]+0.055)/1.055)^2.4
        end
    end
end
end

function processcenterfile(dfcenter, height, width, logfile)
    writecsv(logfile, "") #clear logfile
    setlog = Logger("setlog")
    Logging.configure(setlog, filename=logfile, level=DEBUG)
    
    debug(setlog, "Start calibrate center ")
    calres = calibrate_center(dfcenter, height, width)
    debug(setlog, "calibration result: $calres")
    return (calres)
end

function processprojfile(dfproj, height, width, logfile)
    writecsv(logfile, "") #clear logfile
    setlog = Logger("setlog")
    Logging.configure(setlog, filename=logfile, level=DEBUG)
    
    debug(setlog, "Start calibrate projection ")
    calres = calibrate_projfun(dfproj, height, width)
    debug(setlog, "calibration result: $calres")
    return (calres)
end

function processimages(imagepaths, lensparams, slopeparams, logfile, datafile)
    N = length(imagepaths)

    ## LOGGING
    # Create specific logger per set with debug info
    writecsv(logfile, "") #clear logfile
    setlog = Logger("setlog")
    Logging.configure(setlog, filename=logfile, level=DEBUG)
    
    debug(setlog, "Start `processimages` with lens parameters $lensparams and slope parameters $slopeparams")
    debug(setlog, "received $N image paths")
    
    # create result dictionary
    result = {"success" => false} 
    
    debug(setlog,"create slope object")
    slope, slopeaspect = slopeparams
    if slope == zero(slope)
        myslope = LeafAreaIndex.NoSlope()
    else
        myslope = Slope(slope/180*pi, slopeaspect/180*pi)
    end

    # load first image for image size, required for calibration
    debug(setlog, "load first image for image size from $(imagepaths[1])")
    imgsize = size(readrawjpg(imagepaths[1], myslope))

    debug(setlog, "calibrate CameraLens or load previous calibration")
    mycamlens = load_or_create_CameraLens(imgsize, lensparams, setlog)

    debug(setlog,"parallel process getLAI")
    #needed for anon functions in CameraLens 
    sendto(procs(), lensparams=lensparams, mycamlens=mycamlens, myslope=myslope)
    @everywhere lensx, lensy, lensa, lensb, lensρ = lensparams 
    #remotecall_fetch(2, println, mycamlens)

    resultset = pmap(x->getLAI(x, mycamlens, myslope), imagepaths)
    debug(setlog,"parallel process done")

    # Create datafile with calculated values
    datalog = open(datafile, "w")
    truncate(datalog, 0)
    close(datalog)
    datalog = open(datafile, "a+")
    write(datalog, "Filename, LAI, Threshold_RC, Clumping_LX\n")
    witherror = false
    for lai in resultset
        if !isa(lai, LAIresult)
            witherror = true
            continue
        end
        write(datalog, "$(basename(lai.imagepath)), $(lai.LAI), $(lai.thresh), $(lai.clump)\n")        
    end
    close(datalog)
    witherror && (return result)

    LAIs = Float64[r.LAI for r in resultset]
    result["LAI"] = median(LAIs)
    result["LAIsd"] = StatsBase.mad(LAIs)
    result["success"] = true    
    return(result)
end

function load_or_create_CameraLens(imgsize, lensparams, setlog)
    debug(setlog, "start load_or_create_CameraLens")
    @assert isfile(CAMERALENSES)

    lenshash = string(hash( (imgsize,lensparams) ))  #create unique cameralens identifier
    jld = JLD.jldopen(CAMERALENSES, "r")
    past_hashes = names(jld)
    close(jld)

    if lenshash in past_hashes
        jld = JLD.jldopen(CAMERALENSES, "r")
        debug(setlog, "previous calibration found for hash $lenshash")
        mycamlens = read(jld, lenshash)
        close(jld)
    else
        lensx, lensy, lensa, lensb, lensρ = lensparams
        # Generic functions can't serialize, so need anonymous function to save
        projfθρ = θ -> (lensa*θ + lensb*θ^2) * lensρ
        invprojfρθ = ρ ->(-lensa + sqrt(lensa^2+4lensb*ρ/lensρ)) / 2lensb 
        lensb == zero(lensb) && (invprojfρθ = ρ -> ρ / (lensρ * lensa))
        @assert projfθρ(pi/2) > 2
        @assert projfθρ(pi/2) < maximum(imgsize)

        # Fix likely lens coordinates mistake.
        if lensx > lensy # rowcoord > colcoord
            lensx, lensy = lensy, lensx
            warn("lensx > lensy, probably a mistake, values have been swapped.")
            warn(setlog,"lensx > lensy, probably a mistake, values have been swapped.")   
        end

        debug(setlog,"calibrate new mycamlens")
        mycamlens = CameraLens(imgsize...,lensx,lensy,projfθρ,invprojfρθ)
        debug(setlog,"calibrated new mycamlens, now save to file")
        JLD.jldopen(CAMERALENSES, "r+") do file #"r+" to append writing data
            write(file, lenshash, mycamlens)
        end
        debug(setlog,"new mycamlens saved to file: $lenshash")
    end
    mycamlens
end