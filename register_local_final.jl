using Pkg
Pkg.add("LocalRegistry")
using LocalRegistry

println("--- Attempting Registration via LocalRegistry ---")

# We need to make sure 'General' registry is available locally.
# Pkg.Registry.add("General") usually handles this.

try
    # Try to register to General
    # Note: LocalRegistry usually requires the registry to be 'managed' or creating a new one.
    # But let's see if it allows writing to General if we specify it.
    
    # We must skip the check that registry is dirty, or handle it.
    
    println("Registering to General...")
    register(pwd(), registry="General")
    println("Success! Local General registry updated.")
catch e
    println("Error using LocalRegistry: ", e)
end
