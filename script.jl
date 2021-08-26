using Distributed

#addprocs before including LAIprocessing.jl !
# 9 processors ideal because typical size of image set
addprocs(min(Base.Sys.CPU_THREADS, 9) - nprocs())

using Logging, Dates

# Setup Logging
isdir("logs") || mkdir("logs")
logger_io = open(joinpath("logs", "logjulia_$(Dates.format(Dates.now(), "dd-mm-yyyy_HHhMM")).log"), "a")
logger = SimpleLogger(logger_io, Logging.Debug)
global_logger(logger)
TEMPSETLOG = joinpath("logs", "tempsetlog.log")
DATALOG = joinpath("logs", "data.txt")

#get version of LeafAreaIndex.jl
# using Pkg
# import LibGit2
# LAICOMMIT = LibGit2.readchomp(`rev-parse HEAD`, dir=Pkg.dir("LeafAreaIndex"))

using PyCall, DataFrames
# use Conda.add("pyodbc") if not yet installed
@pyimport pyodbc

include("LAIprocessing.jl")

# check available memory
#import Humanize
#mem() = Humanize.datasize(1000*parse(Int, split(readall(`wmic os get FreePhysicalMemory`))[2]))

# convenience functions for quering the database
function selectcolnames(cursor::PyObject, tablename)
    sql = "SELECT column_name FROM information_schema.columns WHERE table_name = '$tablename'"
    cex = cursor[:execute](sql)
    pytable = cex[:fetchall]()
    String[collect(obj)[1] for obj in pytable]
end
function selecttable(cursor::PyObject, tablename, where::String, justone::Bool)
    # if justone: only select the first valid row
    if justone
        cex = cursor[:execute]("SELECT top 1 * FROM $tablename WHERE $where ORDER BY id ASC")
    else
        cex = cursor[:execute]("SELECT * FROM $tablename WHERE $where ORDER BY id ASC")
    end
    pytable = cex[:fetchall]()
    res = map(collect, pytable)
    df = DataFrame()
    # convert string to Symbol for DataFrame indexing
    colnames = [Symbol(c) for c in selectcolnames(cursor, tablename)]
    for col in eachindex(colnames)
        colvals = [res[row][col] for row in eachindex(res)]
        df[!, colnames[col]] = colvals
    end
    df
end
function selectimages(cursor::PyObject,where::String)
    cex = cursor[:execute]("select path, plotLocations.slope, plotLocations.slopeAspect, images.ID from images left join [dbo].[plotLocations] on images.plotLocationID = [dbo].[plotLocations].ID WHERE $where  order by images.ID ASC")
    pytable = cex[:fetchall]()
    res = map(collect, pytable)
    df = DataFrame()
    # convert string to Symbol for DataFrame indexing
    colnames = ["path","slope","slopeAspect", "ID"]
    for col in eachindex(colnames)
        colvals = [res[row][col] for row in eachindex(res)]
        df[!, colnames[col]] = colvals
    end
    df
end
function updatetable(conn::PyObject, tablename, ID::Int, columnname::Symbol,newvalue)
    cursor = conn[:cursor]()
    sql = "UPDATE $tablename SET $columnname = '$(newvalue)' WHERE ID = $ID"
    cex = cursor[:execute](sql)
    conn[:commit]()
    nothing
end



function process_calibration(conn)

    cursor = conn[:cursor]()
    cameraSetup = selecttable(cursor, :cameraSetup, " processed = 0 and pathCenter is not null ", true)

    size(cameraSetup, 1) != 0 || return
    setupID = cameraSetup[1, :ID]
    try
        @info "$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - detected new processed=false in cameraSetup table"

        #setupID    = cameraSetup[1, :ID]
        pathCenter = cameraSetup[1, :pathCenter]
        pathProj   = cameraSetup[1, :pathProj]
        width      = cameraSetup[1, :width]
        height     = cameraSetup[1, :height]

        df = readtable(pathCenter, names=[:x, :y, :circle])
        lensx, lensy = processcenterfile(df, height, width, TEMPSETLOG)
        @info "$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - result x: $lensx y:$lensy"
        updatetable(conn, "cameraSetup", setupID, :x, lensx)
        updatetable(conn, "cameraSetup", setupID, :y, lensy)

        df = readtable(pathProj, names=[:cm, :px, :H, :pos])
        lensa, lensb = processprojfile(df, height, width, TEMPSETLOG)
        @info "$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - result a: $lensa b:$lensb"
        updatetable(conn, "cameraSetup", setupID, :a, lensa)
        updatetable(conn, "cameraSetup", setupID, :b, lensb)

        updatetable(conn, "cameraSetup", setupID, :processed, 1)
    catch y
        error("Could not process center calibration: $setupID with error $y")
    finally
        updatetable(conn, "cameraSetup", setupID, :processed, 1)
    end
end

function gettabledata(cursor, results)

    @debug "$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - reading LAI_App database tables"

    plotSetID = results[1, :plotSetID];
    @info "$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - plotSetID = $plotSetID"
    plotSet = selecttable(cursor, :plotSets, "ID = $plotSetID", true)
    if size(plotSet)[1] == 0
        error("Could not find plotSetID $plotSetID from results table in plotSets table.")
        set_processed()
        return
    end

    uploadSetID = plotSet[1, :uploadSetID]
    @info "$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - uploadSetID = $uploadSetID"
    uploadSet = selecttable(cursor, :uploadSet, "ID = $uploadSetID", true)
    if size(uploadSet)[1] == 0
        error("Could not find uploadSetID $uploadSetID from results table in uploadSet table.")
        set_processed()
        return
    end

    plotID = plotSet[1, :plotID]
    @info "$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - plotID = $plotID"
    plot = selecttable(cursor, :plots, "ID = $plotID", true)
    if size(plot)[1] == 0
        err("Could not find plotID $plotID from plots table in uploadSet table.")
        set_processed()
        return
    end

    camSetupID = uploadSet[1, :camSetupID]
    @info "$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - camSetupID = $camSetupID"
    cameraSetup = selecttable(cursor, :cameraSetup, "ID = $camSetupID", true)
    if size(cameraSetup)[1] == 0
        error("Could not find camSetupID $camSetupID from uploadSet table in cameraSetup table.")
        set_processed()
        return
    end

    return uploadSet, uploadSetID, plot, plotSetID, cameraSetup
end


function process_images(conn)
    cursor = conn[:cursor]()
    results = selecttable(cursor, "results", "processed = 0", true)
    size(results)[1] != 0 || return
    resultsID = results[1, :ID]
    set_processed() = updatetable(conn, "results", resultsID, :processed, 1)
    # TODO make macro for all these `if not found return` statements
    try
        @info "$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - detected new processed=false in results table"

        uploadSet, uploadSetID, plot, plotSetID, cameraSetup = gettabledata(cursor, results)

        lensx = cameraSetup[1, :x]
        lensy = cameraSetup[1, :y]
        lensa = cameraSetup[1, :a]
        lensb = cameraSetup[1, :b]
        lensρ = cameraSetup[1, :maxRadius]
        lensparams = (lensx, lensy, lensa, lensb, lensρ)
        @info "$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - lens parameters: $lensparams"

        images = selectimages(cursor, "plotSetID = $plotSetID")
        @info "$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - start images processing"
        success = false
        LAIres = Dict()
        try
            LAIres = processimages(images,lensparams,TEMPSETLOG,DATALOG)
            success = LAIres["success"]
            @info "$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - uploadset $uploadSetID process completed with success: $success"
        catch y
            error("$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - Could not process uploadset: $uploadSetID with error $y")
        end
        println("$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - uploadset $uploadSetID process completed with success: $success")
        updatetable(conn, "results", resultsID, :processed, 1)
        updatetable(conn, "results", resultsID, :succes, ifelse(success,1,0))
        updatetable(conn, "results", resultsID, :resultLog, string(read(open(TEMPSETLOG), String)))
        datafile = open(DATALOG)
        updatetable(conn, "results", resultsID, :data, read(datafile, String))
        close(datafile)
        # todo
        LAICOMMIT = "5d043f449684340a392f5df405714dc1b2cbc09f"
        updatetable(conn, "results", resultsID, :scriptVersion, LAICOMMIT)
        if success
            LAIvalue = LAIres["LAI"]
            LAIsd    = LAIres["LAIsd"]
            try
                updatetable(conn, "results", resultsID, :LAI, LAIvalue)
                @info "$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - added LAI to results table for ID $resultsID"
                println("added LAI to results table for ID $resultsID")
                updatetable(conn, "results", resultsID, :LAI_SD, LAIsd)
                @info "$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - added LAI_SD to results table for ID $resultsID"
                for (reskey, col) in [
                    ("csv_gapfraction", :gapfraction), ("csv_exif", :exif),
                    ("csv_histogram", :histogram), ("csv_stats", :stats),
                    ("jpgpath", :jpgPath), ("binpath", :binPath), ("LAIs", :LAI),
                    ("LAIe", :LAIe), ("threshold", :threshold), ("clumping", :clumping),
                    ("overexposure", :overexposure)]
                    for (imgp, csv) in LAIres[reskey]
                        imgp in images.path || continue
                        imageID = images.ID[images.path.==imgp][1]
                        updatetable(conn, "images", imageID , col, csv)
                    end
                end
                for im in eachrow(images)
                    imID = im.ID
                    slope = im.slope
                    slopeaspect = im.slopeAspect
                    updatetable(conn, "images", imID , :slope, slope)
                    updatetable(conn, "images", imID , :slopeAspect, slopeaspect)
                end
            catch y
                error("$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - could not add LAI to results table, error: $y")
            end
        end
    catch y
        error("$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - caugth general error in interior loop: $y")
    finally
        updatetable(conn, "results", resultsID, :processed, 1)
    end
end


# Main loop
function mainloop(conn)
    println("Started Leaf Area Index Service")
    while true
        try
            #process_calibration(conn)
            process_images(conn)
        catch y
            error("$(Dates.format(Dates.now(), "dd u yyyy HH:MM:SS")) - caugth general error in interior loop: $y")
        end
        flush(logger_io)
        sleep(1)
    end
end

# Connect to the database

cnxn  = pyodbc.connect("DSN=LAI")
mainloop(cnxn)
