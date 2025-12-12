using Pkg
Pkg.add("RegistryTools")
using RegistryTools
using LibGit2

println("--- Attempting Registration via RegistryTools ---")

package_dir = pwd()
repo = LibGit2.GitRepo(package_dir)
# Ensure we have a clean state and remote
println("Checking git status...")

# RegistryTools.register(...) might be the command.
# Let's inspect it first in the script to be safe, or just try it.
# Based on online docs for RegistryTools (common name), it might have `register`.

# If RegistryTools.register doesn't exist, we will catch it.
try
    # Printing names to be sure
    println("Exported names in RegistryTools: ", names(RegistryTools))
    
    # Check if register exists
    if isdefined(RegistryTools, :register)
        println("Calling RegistryTools.register...")
        # Note: This usually requires a configured registry, but let's see what it does.
        RegistryTools.register(package_dir)
    else
        println("RegistryTools.register is NOT defined.")
    end
catch e
    println("Error: ", e)
end
