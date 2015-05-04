# LAIscript
scripts to automatically run LAI calculations with ODBC link to custom database

## Installation

Of course, first install [Julia](http://julialang.org/downloads/) and the [LeafAreaIndex package](https://github.com/ETC-UA/LeafAreaIndex.jl) through `Pkg.clone("https://github.com/ETC-UA/LeafAreaIndex.jl")`.

The [Images package](https://github.com/timholy/Images.jl) on Windows might have an issue some versions of the ImageMagick library (that gets automatically installed with the Images package) on reading 16bit (raw) images. In that case, we recommend deleting the folder in de `Images/deps` package folder and install the ImageMagick library separately. Use the `ImageMagick-6.9.1-0-Q16-x64-dll.exe` installer. Finally link to the Images package by running `Pkg.build(Images)`.

Because of a [known bug](https://github.com/quinnj/ODBC.jl/issues/75) with Microsoft SQL, we call the pyodbc Python package through Julia's PyCall. We recommend installing the anaconda Python distribution, which allows to easily install pyodbc in a command window `conda install pyodbc`. 

Besides PyCall and Images, be sure to install the packages Logging, Dates, Humanize and DataFrames

## Run

This script requires a very specific database structure to be present and ODBC in place as can be seen in `script.jl`.

To run, either start julia and `include(script.jl)` , or run `julia script.jl`.

