using System.Diagnostics;

var reader = new BinaryReader(new MemoryStream(File.ReadAllBytes(args[0])));
int leafCount = reader.ReadInt32(), nodeCount = reader.ReadInt32(), problemCount = reader.ReadInt32();
var left = ReadInts(nodeCount);
var right = ReadInts(nodeCount);
var penalty = ReadFloats(nodeCount);
var unaries = new float[problemCount][];
for (var problem = 0; problem < problemCount; problem++)
{
    unaries[problem] = ReadFloats(leafCount);
}

var labels = new byte[problemCount * leafCount];
var total = new double[nodeCount];
var best = new double[nodeCount];
var selected = new bool[nodeCount];
var pending = new int[nodeCount];
SolveAll();
var watch = Stopwatch.StartNew();
SolveAll();
watch.Stop();
File.WriteAllBytes(args[1], labels);
Console.WriteLine($"solved {problemCount} problems n={leafCount} in {watch.Elapsed.TotalMilliseconds:F2} ms");

void SolveAll()
{
    for (var problem = 0; problem < problemCount; problem++)
    {
        Solve(unaries[problem]);
        for (var leaf = 0; leaf < leafCount; leaf++)
        {
            labels[problem * leafCount + leaf] = (byte)(selected[leaf] ? 1 : 0);
        }
    }
}

int[] ReadInts(int count)
{
    var values = new int[count];
    for (var index = 0; index < count; index++)
    {
        values[index] = reader.ReadInt32();
    }
    return values;
}

float[] ReadFloats(int count)
{
    var values = new float[count];
    for (var index = 0; index < count; index++)
    {
        values[index] = reader.ReadSingle();
    }
    return values;
}

void Solve(float[] unary)
{
    for (var leaf = 0; leaf < leafCount; leaf++)
    {
        total[leaf] = unary[leaf];
        best[leaf] = Math.Max(0.0, total[leaf] - penalty[leaf]);
    }
    for (var node = leafCount; node < nodeCount; node++)
    {
        total[node] = total[left[node]] + total[right[node]];
        best[node] = Math.Max(total[node] - penalty[node], best[left[node]] + best[right[node]]);
    }
    Array.Clear(selected);
    var top = 0;
    pending[top++] = nodeCount - 1;
    while (top > 0)
    {
        var node = pending[--top];
        var score = total[node] - penalty[node];
        if (node < leafCount)
        {
            selected[node] = score > 0;
        }
        else if (score > 0 && score >= best[left[node]] + best[right[node]])
        {
            Select(node);
        }
        else
        {
            pending[top++] = left[node];
            pending[top++] = right[node];
        }
    }
}

void Select(int root)
{
    var stack = new Stack<int>();
    stack.Push(root);
    while (stack.Count > 0)
    {
        var node = stack.Pop();
        if (node < leafCount)
        {
            selected[node] = true;
            continue;
        }
        stack.Push(left[node]);
        stack.Push(right[node]);
    }
}
