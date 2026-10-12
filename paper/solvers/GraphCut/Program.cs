using System.Diagnostics;

var reader = new BinaryReader(new MemoryStream(File.ReadAllBytes(args[0])));
int width = reader.ReadInt32(), height = reader.ReadInt32(), problemCount = reader.ReadInt32();
int pixelCount = width * height;
var horizontal = new float[(width - 1) * height];
var vertical = new float[width * (height - 1)];
for (var index = 0; index < horizontal.Length; index++)
{
    horizontal[index] = reader.ReadSingle();
}
for (var index = 0; index < vertical.Length; index++)
{
    vertical[index] = reader.ReadSingle();
}
var unaries = new float[problemCount][];
for (var problem = 0; problem < problemCount; problem++)
{
    unaries[problem] = new float[pixelCount];
    for (var pixel = 0; pixel < pixelCount; pixel++)
    {
        unaries[problem][pixel] = reader.ReadSingle();
    }
}

var labels = new byte[problemCount * pixelCount];
var watch = Stopwatch.StartNew();
var solver = new Dinic(width, height, horizontal, vertical);
for (var problem = 0; problem < problemCount; problem++)
{
    solver.Solve(unaries[problem], labels, problem * pixelCount);
}
watch.Stop();
File.WriteAllBytes(args[1], labels);
Console.WriteLine($"solved {problemCount} problems {width}x{height} in {watch.Elapsed.TotalMilliseconds:F2} ms");

sealed class Dinic
{
    const float Epsilon = 1e-9f;

    readonly int pixelCount;
    readonly int source;
    readonly int sink;
    readonly int nodeCount;
    readonly int[] head;
    readonly int[] next;
    readonly int[] target;
    readonly float[] capacity;
    readonly float[] initialCapacity;
    readonly int[] level;
    readonly int[] cursor;
    readonly int[] queue;
    readonly int[] sourceEdge;
    readonly int[] sinkEdge;
    int edgeCount;

    public Dinic(int width, int height, float[] horizontal, float[] vertical)
    {
        pixelCount = width * height;
        source = pixelCount;
        sink = pixelCount + 1;
        nodeCount = pixelCount + 2;
        var maxEdges = 2 * (pixelCount * 2 + horizontal.Length + vertical.Length);
        head = new int[nodeCount];
        Array.Fill(head, -1);
        next = new int[maxEdges];
        target = new int[maxEdges];
        capacity = new float[maxEdges];
        level = new int[nodeCount];
        cursor = new int[nodeCount];
        queue = new int[nodeCount];
        sourceEdge = new int[pixelCount];
        sinkEdge = new int[pixelCount];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var pixel = y * width + x;
                if (x + 1 < width)
                {
                    var weight = horizontal[y * (width - 1) + x];
                    AddEdge(pixel, pixel + 1, weight, weight);
                }
                if (y + 1 < height)
                {
                    var weight = vertical[y * width + x];
                    AddEdge(pixel, pixel + width, weight, weight);
                }
            }
        }
        for (var pixel = 0; pixel < pixelCount; pixel++)
        {
            sourceEdge[pixel] = edgeCount;
            AddEdge(source, pixel, 0, 0);
            sinkEdge[pixel] = edgeCount;
            AddEdge(pixel, sink, 0, 0);
        }
        initialCapacity = new float[edgeCount];
        Array.Copy(capacity, initialCapacity, edgeCount);
    }

    void AddEdge(int from, int to, float forward, float backward)
    {
        target[edgeCount] = to;
        capacity[edgeCount] = forward;
        next[edgeCount] = head[from];
        head[from] = edgeCount++;
        target[edgeCount] = from;
        capacity[edgeCount] = backward;
        next[edgeCount] = head[to];
        head[to] = edgeCount++;
    }

    public void Solve(float[] unary, byte[] labels, int offset)
    {
        Array.Copy(initialCapacity, capacity, edgeCount);
        for (var pixel = 0; pixel < pixelCount; pixel++)
        {
            if (unary[pixel] > 0)
            {
                capacity[sourceEdge[pixel]] = unary[pixel];
            }
            else if (unary[pixel] < 0)
            {
                capacity[sinkEdge[pixel]] = -unary[pixel];
            }
        }
        while (BuildLevels())
        {
            Array.Copy(head, cursor, nodeCount);
            while (Push(source, float.MaxValue) > Epsilon)
            {
            }
        }
        Reach();
        for (var pixel = 0; pixel < pixelCount; pixel++)
        {
            labels[offset + pixel] = (byte)(level[pixel] >= 0 ? 1 : 0);
        }
    }

    bool BuildLevels()
    {
        Reach();
        return level[sink] >= 0;
    }

    void Reach()
    {
        Array.Fill(level, -1);
        var first = 0;
        var last = 0;
        queue[last++] = source;
        level[source] = 0;
        while (first < last)
        {
            var node = queue[first++];
            for (var edge = head[node]; edge != -1; edge = next[edge])
            {
                if (capacity[edge] > Epsilon && level[target[edge]] < 0)
                {
                    level[target[edge]] = level[node] + 1;
                    queue[last++] = target[edge];
                }
            }
        }
    }

    float Push(int node, float limit)
    {
        if (node == sink)
        {
            return limit;
        }
        for (; cursor[node] != -1; cursor[node] = next[cursor[node]])
        {
            var edge = cursor[node];
            if (capacity[edge] > Epsilon && level[node] < level[target[edge]])
            {
                var pushed = Push(target[edge], MathF.Min(limit, capacity[edge]));
                if (pushed > Epsilon)
                {
                    capacity[edge] -= pushed;
                    capacity[edge ^ 1] += pushed;
                    return pushed;
                }
            }
        }
        return 0;
    }
}
