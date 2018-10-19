using PyCall, DataFrames
using Logging

# use Conda.add("pyodbc") if not yet installed
@pyimport pyodbc

# Setup Logging
!isdir("logs") && mkdir("logs")
Logging.configure(level=DEBUG, filename=joinpath("logs", 
    "logjulia_$(Dates.today())_$(Dates.hour(now()))h$(Dates.minute(now())).log"))
TEMPSETLOG = joinpath("logs", "tempsetlog.log")
DATALOG = joinpath("logs", "data.txt")

#addprocs before including LAIprocessing.jl !
# 9 processors ideal because typical size of image set
addprocs(max(Sys.CPU_CORES, 9) - nprocs())

include("LAIprocessing.jl")

# check available memory
#import Humanize
#mem() = Humanize.datasize(1000*parse(Int, split(readall(`wmic os get FreePhysicalMemory`))[2]))

# convenience functions for quering the database
function selectcolnames(cursor::PyObject, tablename)
    sql = "SELECT column_name FROM information_schema.columns WHERE table_name = '$(tablename)'"
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
    colnames = Symbol[selectcolnames(cursor, tablename)...]
    for col in eachindex(colnames)
        colvals = [res[row][col] for row in eachindex(res)]
        df[colnames[col]] = colvals
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

    try
        info("detected new processed=false in cameraSetup table")
        
        setupID    = cameraSetup[1, :ID]        
        pathCenter = cameraSetup[1, :pathCenter]
        pathProj   = cameraSetup[1, :pathProj]
        width      = cameraSetup[1, :width]
        height     = cameraSetup[1, :height]

        df = readtable(pathCenter, names=[:x, :y, :circle])
        lensx, lensy = processcenterfile(df, height, width, TEMPSETLOG)
        info("result x: $lensx y:$lensy")
        updatetable(conn, "cameraSetup", setupID, :x, lensx)
        updatetable(conn, "cameraSetup", setupID, :y, lensy)

        df = readtable(pathProj, names=[:cm, :px, :H, :pos])
        lensa, lensb = processprojfile(df, height, width, TEMPSETLOG)
        info("result a: $lensa b:$lensb")
        updatetable(conn, "cameraSetup", setupID, :a, lensa)
        updatetable(conn, "cameraSetup", setupID, :b, lensb) 

        updatetable(conn, "cameraSetup", setupID, :processed, 1)
    catch y
        err("Could not process center calibration: $setupID with error $y")
    finally
        updatetable(conn, "cameraSetup", setupID, :processed, 1)       
    end
end

function gettabledata(curs, results)

    debug("reading LAI_App database tables")

    plotSetID = results[1, :plotSetID];
    info("plotSetID = $plotSetID")
    plotSet = selecttable(cursor, :plotSets, "ID = $plotSetID", true)
    if size(plotSet)[1] == 0
        err("Could not find plotSetID $plotSetID from results table in plotSets table.")
        set_processed()
        return
    end

    uploadSetID = plotSet[1, :uploadSetID]
    info("uploadSetID = $uploadSetID")        
    uploadSet = selecttable(cursor, :uploadSet, "ID = $uploadSetID", true)        
    if size(uploadSet)[1] == 0
        err("Could not find uploadSetID $uploadSetID from results table in uploadSet table.")
        set_processed()
        return
    end

    plotID = plotSet[1, :plotID]
    info("plotID = $plotID")
    plot = selecttable(cursor, :plots, "ID = $plotID", true)
    if size(plot)[1] == 0
        err("Could not find plotID $plotID from plots table in uploadSet table.")
        set_processed()
        return
    end
    
    camSetupID = uploadSet[1, :camSetupID]
    info("camSetupID = $camSetupID")        
    cameraSetup = selecttable(cursor, :cameraSetup, "ID = $camSetupID", true)
    if size(cameraSetup)[1] == 0
        err("Could not find camSetupID $camSetupID from uploadSet table in cameraSetup table.")
        set_processed()
        return
    end

    return uploadSet, uploadSetID, plot, plotSetID, cameraSetup
end


function process_images(conn)

    cursor = conn[:cursor]()
    results = selecttable(cursor, "results", "processed = 0", true)
    
    size(results)[1] != 0 || return        

    set_processed() = updatetable(conn, "results", resultsID, :processed, 1)
    # TODO make macro for all these `if not found return` statements
    try
        info("detected new processed=false in results table")

        uploadSet, uploadSetID, plot, plotSetID, cameraSetup = gettabledata(cursor, results)

        lensx = cameraSetup[1, :x]
        lensy = cameraSetup[1, :y]
        lensa = cameraSetup[1, :a]
        lensb = cameraSetup[1, :b]
        lensρ = cameraSetup[1, :maxRadius]
        lensparams = (lensx, lensy, lensa, lensb, lensρ)
        info("lens parameters: $lensparams")

        slope = plot[1, :slope]
        slopeaspect = plot[1, :slopeAspect]
        slopeparams = (slope, slopeaspect)
        info("slope parameters: $slopeparams")

        images = selecttable(cursor, :images, "plotSetID = $plotSetID", false)
        imagepaths = images[:path]

        info("start images processing")
        success = false
        LAIres = Dict()
        try
            LAIres = processimages(imagepaths,lensparams,slopeparams,TEMPSETLOG,DATALOG)                        
            success = LAIres["success"]
            info("uploadset $uploadSetID process completed with success: $success")            
        catch y
            err("Could not process uploadset: $uploadSetID with error $y")
        end

        updatetable(conn, "results", resultsID, :processed, 1)
        updatetable(conn, "results", resultsID, :succes, ifelse(success,1,0))
        updatetable(conn, "results", resultsID, :resultLog, string(readall(open(TEMPSETLOG))))
        datafile = open(DATALOG)
        updatetable(conn, "results", resultsID, :data, readall(datafile))
        close(datafile)
        if success
            LAIvalue = LAIres["LAI"]
            LAIsd    = LAIres["LAIsd"]
            try
                updatetable(conn, "results", resultsID, :LAI, LAIvalue)
                info("added LAI to results table for ID $resultsID")
                updatetable(conn, "results", resultsID, :LAI_SD, LAIsd)
                info("added LAI_SD to results table for ID $resultsID")        
            catch upy
                err("could not add LAI to results table, error: $y")
            end
        end
    catch y
        err("caugth general error in interior loop: $y")
    finally
        updatetable(conn, "results", resultsID, :processed, 1)
    end
end


# Main loop
function mainloop(conn)
    println("Started Leaf Area Index Service")
    while true
        try 
            process_calibration(conn)
            process_images(conn)
        catch y
            err("caugth general error in interior loop: $y")
        end 
        sleep(1)
    end
end

# Connect to the database

#cnxn  = pyodbc.connect("DSN=LAI")
#mainloop(cnxn)
