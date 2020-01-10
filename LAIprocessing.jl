using Logging, LeafAreaIndex, ParallelDataTransfer, CoordinateTransformations
using PyCall
# TODO use MicroLogging on julia 0.6 or Base.Logging on julia 1.0
using Images #also imports FileIO for reading jpg
import StatsBase, JLD2, FileIO

CAMERALENSES = "CameraLenses.jld2"
if !isfile(CAMERALENSES)
    warn("file with previous CameraLens calibrations not found, will create empty one called $CAMERALENSES")
    close(JLD2.jldopen(CAMERALENSES,"w"))
end

@everywhere begin
    Lg = Logging
	
    abstract type LAIresultInfo; end
    "Convenience type to hold results from LAI calculation."
    type LAIresult <: LAIresultInfo
        imagepath::AbstractString
        LAI::Float64
        LAIe::Float64
        thresh::Float64
        clump::Float64
        overexposure::Float64
        csv_gapfraction::AbstractString
        csv_histogram::AbstractString
        csv_exif::AbstractString
        csv_stats::AbstractString
    end
    type NoLAIresult <: LAIresultInfo
        exception::Exception
    end

    function getLAI(imagepath::AbstractString, cl::LeafAreaIndex.CameraLens, 
                    sl::LeafAreaIndex.SlopeInfo)
        # This function gets executed in parallel, so need to set up new logger
        # on each processor.
		id = myid() # ID of current processor for logging file
        @show "before logger"
        #baselog, logext = splitext(mainlogfile)
        #locallogfile = baselog * string(myid()) * logext
        #writecsv(locallogfile, "") #clear logfile
        #println("csv written")
		@show locallog = Lg.Logger("locallog")
		locallogfile = joinpath("logs", "locallog$(id).log")
		Lg.configure(locallog, filename=locallogfile, level=Lg.DEBUG)
        try
            #@show ("start getLAI on $imagepath")
			Lg.debug(locallog, "start processing on $imagepath")
            img = readrawjpg(imagepath, sl)            
            #@show "image read"
            Lg.debug(locallog, "image read")
            polim = LeafAreaIndex.PolarImage(img, cl, sl)
            Lg.debug(locallog, "PolarImage created")
            thresh = LeafAreaIndex.threshold(polim)
			@show "thresh calculated"
			Lg.debug(locallog, "threshold calculated")
            csv_gf = csv_gapfraction(polim, thresh)
			@show "csv_gapfraction calculated"
            Lg.debug(locallog, "created csv_gapfraction" )
			csv_hist = csv_histogram(polim.img)
            csv_ex = csv_exif(imagepath)
            csv_st = csv_stats(polim)
            write_bin_jpg(polim, thresh, imagepath)
            Lg.debug(locallog,"wrote bin and jpg")        
            LAIe = LeafAreaIndex.inverse(polim, thresh)
            Lg.debug(locallog,"effective LAI: $LAIe")
            clump = LeafAreaIndex.langxiang45(polim, thresh, 0, pi/2)
            Lg.debug(locallog,"clumping: $clump")
            LAI = LAIe / clump
            Lg.debug(locallog, "LAI: $LAI")   
            overexposure = sum(img .== 1) / (pi * cl.fθρ(pi/2)^2)
			Lg.debug(locallog, "overexposure calculated")
			res = LAIresult(imagepath, LAI, LAIe, thresh, clump, overexposure, csv_gf, csv_hist, csv_ex, csv_st)
			Lg.debug(locallog, "LAIresult created")
            return res
        catch lai_err
            Lg.debug(locallog, "error: $lai_err")
            #@show lai_err
            return NoLAIresult(lai_err)
        end
    end

    function readrawjpg(imp::AbstractString, sl::LeafAreaIndex.SlopeInfo)
        #i = myid()# ID of current processor
        #debug(setlog, "$i start reading $imp")
        @assert imp != nothing
        @assert isfile(imp)

        ext = lowercase(splitext(imp)[end])

        if ext in LeafAreaIndex.RAW_EXT
            imgblue = LeafAreaIndex.rawblueread(imp)        
        elseif ext in [".jpg",".jpeg", ".tiff"]
            img = FileIO.load(imp)
            imgblue = Images.blue.(img)
            gamma_decode!(imgblue)
        else
            warn("image has unknown extension at $imp")
            #warn(setlog,"$i image has unknown extension at $imp")
        end
        #debug(setlog, "image read")

        #@show "check overexposure"
        if sum(imgblue .== 1) > 0.005 * length(imgblue)
            warn("Image overexposed: $imp")
            #warn(setlog, "$i Image overexposed: $imp")
        end

        #rotate if in portrait mode
        if size(imgblue,1) > size(imgblue,2)
            slope = LeafAreaIndex.params(sl)[1]
            if slope != zero(slope)                
                #warn(setlog, "$i image with slope in portrait mode, don't know which way to turn: $imp")
                error("image with slope in portrait mode, don't know which way to turn: $imp")
            end
            imgblue = rotate90(imgblue) #default clockwise, could influence result due to lens center
        end
        #println("return rrj")
        return imgblue
    end

    # Rotate sometimes because currently LeafAreaIndex expects landscape *in memory*.
    "Rotates an image (or in general an `AbstractMatrix`) 90 degrees."    
    function rotate90(img; clockwise=true)
        transf = recenter(RotMatrix(ifelse(clockwise, 1, -1)*pi/2), center(img)) 
        img = warp(img, transf)
        #fix for images.jl #717
        return parent(img)
    end

    "Gamma decode a gray image taken from single channel in sRGB colorspace."
    function gamma_decode!(A::AbstractMatrix)
        @fastmath for i in eachindex(A)
            # See https://en.wikipedia.org/wiki/SRGB
            A[i] = A[i] <= 0.04045 ? A[i]/12.92 : ((A[i]+0.055)/1.055)^2.4
        end
    end

    function write_bin_jpg(polim::LeafAreaIndex.PolarImage, thresh, imgfilepath)
        jpgfilepath, binfilepath = make_bin_jpg_paths(imgfilepath)
        left, right, down, up = cropbox(polim)
        
        image = polim.img[down:up, left:right]
        Images.save(jpgfilepath, image)
        imgage_gray = Images.Gray.(image .> thresh)
        Images.save(binfilepath, imgage_gray)
        return
    end
    function cropbox(polim::LeafAreaIndex.PolarImage)
        radius = floor(Int, polim.cl.fθρ(pi/2))
        left, right = polim.cl.cj - radius, polim.cl.cj + radius
        down, up    = polim.cl.ci - radius, polim.cl.ci + radius
        # prevent out of bounds
        left, right  = max(1, left), min(size(polim.cl)[2], right)
        down, up     = max(1, down), min(size(polim.cl)[1], up)
        return left, right, down, up
    end
    function make_bin_jpg_paths(imgfilepath)
        imgdir, imgfile = splitdir(imgfilepath)
        imgbase, imgext = splitext(imgfile)
        bindir = joinpath(imgdir, "bin")
        isdir(bindir) || mkpath(bindir)
        binfilepath = joinpath(bindir, imgbase*"_bin.png")
        jpgdir = joinpath(imgdir, "jpg")
        isdir(jpgdir) || mkpath(jpgdir)
        jpgfilepath = joinpath(jpgdir, imgbase*"_jpg.jpg")
        return jpgfilepath, binfilepath
    end

    function csv_gapfraction(polim::LeafAreaIndex.PolarImage, thresh)
        Nrings = LeafAreaIndex.Nrings_def(polim)
        θmax = LeafAreaIndex.maxviewangle(polim)
        θedges, θmid, K = LeafAreaIndex.contactfreqs(polim, 0.0, θmax, Nrings, thresh)
        T = exp.(-K./cos.(θmid))
        csv = "view_angle, gapfraction\n "
        for i in 1:length(T)
            csv *= "$(θmid[i]), $(T[i])\n "
        end
        return csv
    end

    function csv_histogram(img, bins=256)
        hist_range = -1 / (bins-1) : 1 / (bins-1) : 1
        log10nz(x) = x == 0 ? 0 : log10(x)
        counts = log10nz.(LeafAreaIndex.fasthist(reshape(img, length(img)), hist_range))
        csv = "intensity, count_log10\n "
        for i in 1:length(counts)
            csv *= "$(i/bins), $(counts[i])\n "
        end
        return csv
    end

    PyCall.py"""
    import exifread

    def exif(path):
        tags = {}
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        kinds = ['EXIF ExposureTime', 'EXIF FNumber', 'EXIF ISOSpeedRatings', 'Image Orientation', 
                'Image DateTime', "Image Make", "Image Model", "EXIF FocalLength" ]
        # for kind in kinds:
        #     print(str(tags.get(kind)))
        res = {k:str(v) for (k,v) in tags.items() if k in kinds}
        return res
    """
    function csv_exifs(imgfilepath)
        tags = PyCall.py"exifs"(file)
        csv = "key, value\n "
        for (k,v) in tags
            csv *= "$k, $v\n "
        end
        return csv
    end
    function csv_stats(polim::LeafAreaIndex.PolarImage)
        left, right, down, up = cropbox(polim)
        image = polim.img[down:up, left:right]
        len = length(image)
        image = reshape(image, len)
        sort!(image)
        csv = "percentile, intensity\n "
        for p in [0.95, 0.98, 0.99, 0.999]
            csv *= "$(p), $(float(image[floor(Int, len*p)])) \n "
        end
        csv *= "max, $(float(image[end])) \n "
        return csv
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
    #writecsv(logfile, "") #clear logfile
    setlog = Logger("setlog")
    Logging.configure(setlog, filename=logfile, level=DEBUG)
    
    debug(setlog, "Start `processimages` with lens parameters $lensparams and slope parameters $slopeparams")
    debug(setlog, "received $N image paths")
    
    # create result dictionary
    result = Dict{String, Any}("success" => false)
    
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

    debug(setlog ,"parallel process getLAI")
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
    write(datalog, "Filename, LAI, LAIe, Threshold_RC, Clumping_LX, Overexposure\n")
    witherror = false
    result["csv_gapfraction"] = Dict{String, String}()
    for lai in resultset
        if !isa(lai, LAIresult)
            witherror = true
            debug(setlog, "found error in LAIresult $lai")
            continue
        end
        overexp_str = @sprintf("%.7f", lai.overexposure)
        write(datalog, "$(basename(lai.imagepath)), $(lai.LAI), $(lai.LAIe), $(lai.thresh), $(lai.clump), $(overexp_str)\n")
        result["csv_gapfraction"][lai.imagepath] = lai.csv_gapfraction
        result["csv_histogram"][lai.imagepath] = lai.csv_histogram
        result["csv_exif"][lai.imagepath] = lai.csv_exif
        result["csv_stats"][lai.imagepath] = lai.csv_stats
    end
    close(datalog)
    debug(setlog,"closed $datafile")
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
    past_hashes = JLD2.jldopen(CAMERALENSES, "r") do file
         keys(file)
    end

    if lenshash in past_hashes
        debug(setlog, "previous calibration found for hash $lenshash")
        mycamlens = FileIO.load(CAMERALENSES, lenshash)
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
        JLD2.jldopen(CAMERALENSES, "r+") do file #"r+" to append writing data
            file[lenshash] = mycamlens
        end
        debug(setlog,"new mycamlens saved to file: $lenshash")
    end
    mycamlens
end
