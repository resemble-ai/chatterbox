using CSnakes.Runtime;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System.CommandLine;

var home = Path.Join(Environment.CurrentDirectory, "."); /* Path to your Python modules */

var builder = Host.CreateDefaultBuilder(args)
    .ConfigureServices(services =>
    {
        services
            .WithPython()
            .WithHome(home)
            // Download Python 3.12 and store it locally
            .FromRedistributable()
            // Add these methods to configure a virtual environment and install packages from requirements.txt
            .WithVirtualEnvironment(Path.Join(home, ".venv"))
            .WithPipInstaller();
    });

var app = builder.Build();
var env = app.Services.GetRequiredService<IPythonEnvironment>();

var rootCommand = new RootCommand("Reads out websites and text files");
rootCommand.Add(new ReadCommand(env));
rootCommand.Add(new DumpCommand());
rootCommand.Add(new ConcatOutputCommand(env));
await rootCommand.InvokeAsync(args);

