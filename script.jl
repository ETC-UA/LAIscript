using PyCall, DataFrames

using Logging
using Dates, Humanize

# Setup Logging
!isdir("logs") && mkdir("logs")
Logging.configure(level=DEBUG, 
    filename=joinpath("logs","logjulia_$(today())_$(hour(now()))h$(minute(now())).log"))
tempsetlog = joinpath("logs", "tempsetlog.log")

include("LAIprocessing.jl")

# check available memory
mem() = datasize(1000*int(split(readall(`wmic os get FreePhysicalMemory`))[2]))

# convenience functions for quering the database
function selectcolnames(cursor::PyObject, tablename::ASCIIString)
    sql = "SELECT column_name FROM information_schema.columns WHERE table_name = '$(tablename)'"
    cex = cursor[:execute](sql)
    pytable = cex[:fetchall]()    
    ASCIIString[collect(obj)[1] for obj in pytable]
end
function selecttable(cursor::PyObject, tablename::ASCIIString)
    cex = cursor[:execute]("SELECT * FROM $tablename")
    pytable = cex[:fetchall]()
    res = [collect(pytable[i]) for i=1:length(pytable)]
    df = DataFrame()
    # convert string to Symbol for DataFrame indexing
    colnames = Symbol[selectcolnames(cursor, tablename)...]
    for col = 1:length(colnames)
        colvals = [res[row][col] for row = 1:length(res)]
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
    results = selecttable(cursor, "results")
    
    if any(results[:processed] .== false)
        resultsID = results[findfirst(results[:processed], false), :ID]
        try
            info("detected new processed=false in results table")

            debug("reading LAI_App database tables")
            uploadSet = selecttable(cursor, "uploadSet")
            cameraSetup = selecttable(cursor, "cameraSetup")
            images = selecttable(cursor, "images")
            
            uploadSetID = results[findfirst(results[:processed], false), :uploadSetID]
            info("uploadSetID = $uploadSetID")

            uploadSetrow = findfirst(uploadSet[:ID],uploadSetID)
            if uploadSetrow == 0
                err("Could not find uploadSetID $uploadSetID from results table in uploadSet table.")
                updatetable(conn, "results", resultsID, :processed, 1)
                return()
            end
            camSetupID = uploadSet[uploadSetrow, :camSetupID]
            info("camSetupID = $camSetupID")

            camrow = findfirst(cameraSetup[:ID], camSetupID)
            if camrow == 0
                err("Could not find camSetupID $camSetupID from uploadSet table in cameraSetup table.")
                updatetable(conn, "results", resultsID, :processed, 1)
                return()
            end
            lensx = cameraSetup[camrow, :x]
            lensy = cameraSetup[camrow, :y]
            lensa = cameraSetup[camrow, :a]
            lensb = cameraSetup[camrow, :b]
            imagepaths = images[images[:setID] .== uploadSetID, :dngPath]

            info("start images processing")
            success = false
            LAIres = Dict()
            try
                LAIres = processimages(imagepaths,lensx,lensy,lensa,lensb,tempsetlog)                        
                success = LAIres["success"]
                info("uploadset $uploadSetID process completed with success: $success")            
            catch y
                err("Could not process uploadset: $uploadSetID with error $y")
            end

            updatetable(conn, "results", resultsID, :processed, 1)
            updatetable(conn, "results", resultsID, :succes, ifelse(success,1,0))
            updatetable(conn, "results", resultsID, :resultLog, string(readall(open(tempsetlog))))

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
#mainloop(cnxn)
