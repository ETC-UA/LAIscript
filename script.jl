using PyCall, DataFrames
using Logging

# Setup Logging
!isdir("logs") && mkdir("logs")
Logging.configure(level=DEBUG, filename=joinpath("logs", 
    "logjulia_$(Dates.today())_$(Dates.hour(now()))h$(Dates.minute(now())).log"))
tempsetlog = joinpath("logs", "tempsetlog.log")
datalog = joinpath("logs", "data.txt")

#addprocs before including LAIprocessing.jl !
# 9 processors ideal because typical size of image set
addprocs(max(CPU_CORES, 9) - nprocs())

include("LAIprocessing.jl")

# check available memory
#import Humanize
#mem() = Humanize.datasize(1000*parse(Int, split(readall(`wmic os get FreePhysicalMemory`))[2]))

# convenience functions for quering the database
function selectcolnames(cursor::PyObject, tablename::ASCIIString)
    sql = "SELECT column_name FROM information_schema.columns WHERE table_name = '$(tablename)'"
    cex = cursor[:execute](sql)
    pytable = cex[:fetchall]()
    ASCIIString[collect(obj)[1] for obj in pytable]
end
function selecttable(cursor::PyObject, tablename::ASCIIString, where::ASCIIString, justone::Bool)
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
function updatetable(conn::PyObject, tablename::ASCIIString, ID::Int, columnname::Symbol,newvalue)
    cursor = conn[:cursor]()
    sql = "UPDATE $tablename SET $columnname = '$(newvalue)' WHERE ID = $ID"
    cex = cursor[:execute](sql)
    conn[:commit]()
    nothing
end

# Connect to the database
@pyimport pyodbc
cnxn  = pyodbc.connect("DSN=LAI")

#loop interior (separate function for testing)
function loopinterior(conn)

    cursor = conn[:cursor]()

    cameraSetup = selecttable(cursor, "cameraSetup", " processed = 0 and pathCenter is not null ", true)
    
    if size(cameraSetup)[1] != 0
        setupID = cameraSetup[1, :ID]
        try
            info("detected new processed=false in cameraSetup table")
            
            pathCenter = cameraSetup[1, :pathCenter]
            pathProj = cameraSetup[1, :pathProj]
            width = cameraSetup[1, :width]
            height = cameraSetup[1, :height]

            success = false
            LAIres = Dict()
            try
                df = readtable(pathCenter, names=[:x, :y, :circle])
                lensx, lensy = processcenterfile(df, height, width, tempsetlog)
                info("result x: $lensx y:$lensy")
                updatetable(conn, "cameraSetup", setupID, :x, lensx)
                updatetable(conn, "cameraSetup", setupID, :y, lensy)
                df = readtable(pathProj, names=[:cm, :px, :H, :pos])
                lensa, lensb = processprojfile(df, height, width, tempsetlog)
                info("result a: $lensa b:$lensb")
                updatetable(conn, "cameraSetup", setupID, :a, lensa)
                updatetable(conn, "cameraSetup", setupID, :b, lensb)                
            catch y
                err("Could not process center calibration: $setupID with error $y")
            end

            updatetable(conn, "cameraSetup", setupID, :processed, 1)
            
            
        catch y
            err("caugth general error in interior loop: $y")
        finally
            updatetable(conn, "cameraSetup", setupID, :processed, 1)
        end
    else
    end




    results = selecttable(cursor, "results", "processed = 0", true)
    
    if size(results)[1] != 0
        resultsID = results[1, :ID]
        try
            info("detected new processed=false in results table")

            debug("reading LAI_App database tables")
            plotSetID = results[1, :plotSetID]
            info("plotSetID = $plotSetID")

            plotSet = selecttable(cursor, "plotSets", "ID = $plotSetID", true)
            if size(plotSet)[1] == 0
                err("Could not find plotSetID $plotSetID from results table in plotSets table.")
                updatetable(conn, "results", resultsID, :processed, 1)
                return
            end

            uploadSetID = plotSet[1, :uploadSetID]
            info("uploadSetID = $uploadSetID")
            
            uploadSet = selecttable(cursor, "uploadSet", "ID = $uploadSetID", true)
            
            if size(uploadSet)[1] == 0
                err("Could not find uploadSetID $uploadSetID from results table in uploadSet table.")
                updatetable(conn, "results", resultsID, :processed, 1)
                return
            end

            plotID = plotSet[1, :plotID]
            info("plotID = $plotID")

            plot = selecttable(cursor, "plots", "ID = $plotID", true)

            if size(plot)[1] == 0
                err("Could not find plotID $plotID from plots table in uploadSet table.")
                updatetable(conn, "results", resultsID, :processed, 1)
                return
            end
            
            camSetupID = uploadSet[1, :camSetupID]
            info("camSetupID = $camSetupID")
            
            cameraSetup = selecttable(cursor, "cameraSetup", "ID = $camSetupID", true)
            if size(cameraSetup)[1] == 0
                err("Could not find camSetupID $camSetupID from uploadSet table in cameraSetup table.")
                updatetable(conn, "results", resultsID, :processed, 1)
                return
            end

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

            images = selecttable(cursor, "images", "plotSetID = $plotSetID", false)
            imagepaths = images[:path]

            info("start images processing")
            success = false
            LAIres = Dict()
            try
                LAIres = processimages(imagepaths,lensparams,slopeparams,tempsetlog,datalog)                        
                success = LAIres["success"]
                info("uploadset $uploadSetID process completed with success: $success")            
            catch y
                err("Could not process uploadset: $uploadSetID with error $y")
            end

            updatetable(conn, "results", resultsID, :processed, 1)
            updatetable(conn, "results", resultsID, :succes, ifelse(success,1,0))
            updatetable(conn, "results", resultsID, :resultLog, string(readall(open(tempsetlog))))
            datafile = open(datalog)
            updatetable(conn, "results", resultsID, :data, readall(datafile))
            close(datafile)
            if success
                LAIvalue = LAIres["LAI"]
                LAIsd = LAIres["LAIsd"]
                try
                    updatetable(conn, "results", resultsID, :LAI, LAIvalue)
                    info("added LAI to results table for ID $resultsID")
                    updatetable(conn, "results", resultsID, :LAIsd, LAIsd)
                    info("added LAIsd to results table for ID $resultsID")        
                catch upy
                    err("could not add LAI to results table, error: $y")
                end
            end
        catch y
            err("caugth general error in interior loop: $y")
        finally
            updatetable(conn, "results", resultsID, :processed, 1)
        end
    else
        sleep(1)
    end
    nothing
end

# Main loop
function mainloop(conn)
    println("Started Leaf Area Index Service")
    while true
        loopinterior(conn)
    end
end
mainloop(cnxn)
