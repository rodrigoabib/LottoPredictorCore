using System;

public class MegasenaResult
{
    public int V1 { get; private set; }
    public int V2 { get; private set; }
    public int V3 { get; private set; }
    public int V4 { get; private set; }
    public int V5 { get; private set; }
    public int V6 { get; private set; }
    
    
    public MegasenaResult(int v1, int v2, int v3, int v4, int v5, int v6)
    {
        V1 = v1;
        V2 = v2;
        V3 = v3;
        V4 = v4;
        V5 = v5;
        V6 = v6;
    }

    public MegasenaResult(double[] values)
    {
        V1 = (int)Math.Round(values[0]);
        V2 = (int)Math.Round(values[1]);
        V3 = (int)Math.Round(values[2]);
        V4 = (int)Math.Round(values[3]);
        V5 = (int)Math.Round(values[4]);
        V6 = (int)Math.Round(values[5]);
    }

    public bool IsValid()
    {
        return
        V1 >= 1 && V1 <= 60 &&
        V2 >= 1 && V2 <= 60 &&
        V3 >= 1 && V3 <= 60 &&
        V4 >= 1 && V4 <= 60 &&
        V5 >= 1 && V5 <= 60 &&
        V6 >= 1 && V6 <= 60 &&
        V1 != V2 &&
        V1 != V3 &&
        V1 != V4 &&
        V1 != V5 &&
        V1 != V6 &&
        V2 != V3 &&
        V2 != V4 &&
        V2 != V5 &&
        V2 != V6 &&
        V3 != V4 &&
        V3 != V5 &&
        V3 != V6 &&
        V4 != V5 &&
        V4 != V6 &&
        V5 != V6;
    }

    public bool IsOut()
    {
        return
        !(
        V1 >= 1 && V1 <= 60 &&
        V2 >= 1 && V2 <= 60 &&
        V3 >= 1 && V3 <= 60 &&
        V4 >= 1 && V4 <= 60 &&
        V5 >= 1 && V5 <= 60 &&
        V6 >= 1 && V6 <= 60);
    }

    public override string ToString()
    {
        return string.Format(
        "{0},{1},{2},{3},{4},{5}",
        V1, V2, V3, V4, V5, V6);
    }
}