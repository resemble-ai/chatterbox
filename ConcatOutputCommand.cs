using System.CommandLine;
using CSnakes.Runtime;
public class ConcatOutputCommand : Command
{
    private readonly Argument<string> prefixOption = new("--prefix", () => "output", "Source folder for *.wav files");

    private readonly Option<DirectoryInfo> sourceOption = new("--source", () => new DirectoryInfo(Environment.CurrentDirectory), "Source folder for *.wav files");

    private readonly Option<FileInfo> outputOption = new("--output", () => new FileInfo(Path.Combine(Environment.CurrentDirectory, "output.mp3")), "Output mp3 path");

    private readonly IPythonEnvironment environment;

    public ConcatOutputCommand(IPythonEnvironment environment) : base("concat", "Concatenates output .wav audio files into a single .mp3.")
    {
        this.environment = environment;
        this.Add(this.prefixOption);
        this.Add(this.sourceOption);
        this.Add(this.outputOption);
        this.SetHandler(this.Concatenate, this.sourceOption, this.prefixOption, this.outputOption);
    }

    public void Concatenate(DirectoryInfo sourceFolder, string prefix, FileInfo outputFile)
    {
        var module = this.environment.GenerateWithVoice();
        var files = sourceFolder.EnumerateFiles($"{prefix}-*.wav", SearchOption.TopDirectoryOnly)
            .OrderBy(p => Path.GetFileNameWithoutExtension(p.Name))
            .Select(f => f.FullName)
            .ToList();
        Console.WriteLine($"Writing {files.Count} wave files into {outputFile.Name}");
        module.ConcatenateWavFilesSafe(files, outputFile.FullName);
        Console.WriteLine($"Done");
    }
}