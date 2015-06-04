using Images
using Logging
using LeafAreaIndex
import StatsBase


# using Debug
# @debug 
function processimages(imagepaths,lensx,lensy,lensa,lensb,slope,slopeaspect,logfile,datafile)

    # Create specific logger per set with debug info
    writecsv(logfile,"") #clear logfile
    setlog = Logger("setlog")
    Logging.configure(setlog, filename=logfile, level=DEBUG)
    # Create datafile with calculated values and clear it
    datalog = open(datafile,"w")
    truncate(datalog,0)
    close(datalog)
    datalog = open(datafile,"a+")
    write(datalog,"filename;Ridler Calvard threshold;Lang Xiang clumping;LAI with optimization;Edge Detection threshold;Lang Xiang clumping;LAI with optimization\n")
    
    debug(setlog,"Start `processimages` with lensx=$lensx, lensy=$lensy, lensa=$lensa and lensb=$lensb")
    debug(setlog,"recieved $(length(imagepaths)) image paths")
    
    # create result dictionary
    result = {"success" => false} 
    
    imgs = Matrix[]
    #waited_once = false #give more time to DNG converter
    for imp in imagepaths
        
        debug(setlog,"start reading $imp")
        if imp == nothing 
            warn("could not read empty path")
            warn(setlog,"could not read empty path")
            continue
        end
        
        if !isfile(imp) #&& !waited_once
            #waited_once = true
            #sleep(10)
            sleep(1)
        end

        if !isfile(imp)
            warn("could not open image path $imp")
            warn(setlog,"could not open image path $imp")
            continue
        end
        
        img = imread(imp)
        debug("image read")
        imgarray = convert(Array,img)
        imgblue = Images.blue(img)
        if sum(imgblue .== 1) > 0.005 * length(imgblue)
            warn("Image overexposed: $imp")
            warn(setlog, "Image overexposed: $imp")
        end
        #rotate if in landscape mode
        if size(imgblue,1) > size(imgblue,2); imgblue = transpose(imgblue);end
        push!(imgs,imgblue)
    end
        
    if isempty(imgs)
        err("could not read any image, process will exit")
        err(setlog,"could not read any image, process will exit")
        return(result)
    end
    if any(size(imgs[1]) .!= [size(i) for i in imgs])
        err("not all images are the same size, process will exit")
        return(result)
    end
    
    debug(setlog,"calibrate camera")
    projfθρ(θ) = lensa*θ + lensb*θ^2
    if projfθρ(pi/2) < 2
        err("Projection function for π/2 <2, so probably forgot to include maximum radius.")
        err(setlog,"Projection function for π/2 <2, so probably forgot to include maximum radius.")
        return(result)
    end
    if projfθρ(pi/2) > 2(maximum(size(imgs[1]))/2)
        warn("Projection function for π/2 >2*radius, probably a mistake.")
        warn(setlog,"Projection function for π/2 >2*radius, probably a mistake.")        
    end
    invprojfρθ(ρ) = (-lensa + sqrt(lensa^2+4lensb*ρ)) / 2lensb
    
    # Fix likely lens coordinates mistake.
    if lensx < lensy  
        lensx, lensy = lensy, lensx
    end
    lensi=lensy
    lensj=lensx

    mycamlens = calibrate(size(imgs[1],1),size(imgs[1],2),lensi,lensj,projfθρ,invprojfρθ)
    
    debug(setlog,"create PolarImages")
    if slope != zero(slope) 
        myslope = Slope(slope/180*pi, slopeaspect/180*pi)
        polimgs = [PolarImage(img, mycamlens, myslope) for img in imgs]
    else
        polimgs = [PolarImage(img, mycamlens) for img in imgs]
    end    
    
    # @bp
    LAIs = Float64[] #list of LAI
    function pushLAI!(LAIs, polim, th)
        clump = langxiang45(polim, th, 0, pi/2)
        debug(setlog,"Lang Xiang clumping: $clump")
        write(datalog,"$clump;")
        LAI = ellips_LUT(polim, th) / clump
        debug(setlog, "LAI with LUT: $LAI")
        push!(LAIs, LAI)
        LAI = ellips_opt(polim, th) / clump
        debug(setlog, "LAI with optimization: $LAI")
        write(datalog,"$LAI")
        push!(LAIs, LAI)
    end
    
    counter = 0
    for pim in polimgs
        counter += 1

        debug(setlog,"Processing with Ridler Calvard for image $counter")
        write(datalog,"image $counter;")
        th_RC = RidlerCalvard(pim)
        debug(setlog,"Ridler Calvard threshold: $th_RC")
        write(datalog,"$th_RC;")
        pushLAI!(LAIs, pim, th_RC)
        write(datalog,";")
        try # Minimum threshold does not always converge
            debug(setlog,"Processing with Minimum threshold for image $counter")
            th_min = minimum_threshhold(pim)
            debug(setlog,"Minimum threshold: $th_min")
            pushLAI!(LAIs, pim, th_min)
        end

        debug(setlog, "Processing with Edge Detection for image $counter")
        th_edge = edge_threshold(pim)
        debug(setlog, "Edge Detection threshold for image $counter: $th_edge")
        write(datalog,"$th_edge;")
        pushLAI!(LAIs, pim, th_edge)
        write(datalog,"\n")
    end
    
    result["LAI"] = median(LAIs)
    result["LAIsd"] = StatsBase.mad(LAIs)
    result["success"] = true    
    close(datalog)
    return(result)
end
