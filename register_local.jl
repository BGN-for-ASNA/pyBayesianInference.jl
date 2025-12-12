using Pkg
using Registrator

println("--- Starting Registration ---")

# Define the package and repo
# We assume the current directory is the package
package_dir = pwd()
repo_url = "https://github.com/BGN-for-ASNA/BIJ" 

println("Package Directory: ", package_dir)
println("Repo URL: ", repo_url)

reg = Registrator.RegEdit(
    package="BayesianInference", 
    repo=repo_url
)

println("Submitting registration request...")
try
    Registrator.register(reg)
    println("Registration request submitted successfully!")
catch e
    println("Error during registration:")
    println(e)
end
