typedef float ScalarType;
#define max_edge_num 50

__device__ void intersection( ScalarType* s,  ScalarType* e, ScalarType *clipsquare, const int& flag, ScalarType* intersect)
{
    // 0,1,2,3 分别代表 上，右，下，左 （如果y轴向下，x轴向右）

    switch(flag)
    {
        case 0:
            intersect[0] = (s[0] - e[0]) * (clipsquare[1] - e[1]) / (s[1] - e[1]) + e[0];
            intersect[1] = ScalarType(clipsquare[1]);
            break;
        case 1:
            intersect[1] = (s[1] - e[1]) * (clipsquare[2] - e[0]) / (s[0] - e[0]) + e[1];
            intersect[0] = ScalarType(clipsquare[2]);
            break;
        case 2:
            intersect[0] = (s[0] - e[0]) * (clipsquare[3] - e[1]) / (s[1] - e[1]) + e[0];
            intersect[1] = ScalarType(clipsquare[3]);
            break;
        case 3:
            intersect[1] = (s[1] - e[1]) * (clipsquare[0] - e[0]) / (s[0] - e[0]) + e[1];
            intersect[0] = ScalarType(clipsquare[0]);
            break;
        default:
            break;
    }
}

__device__ bool inside( ScalarType* p_xy, ScalarType *clipsquare, int& flag)
{
    switch(flag)
    {
        case 0:
            return (p_xy[1] > clipsquare[1]);
        case 1:
            return (p_xy[0] < clipsquare[2]);
        case 2:
            return (p_xy[1] < clipsquare[3]);
        case 3:
            return (p_xy[0] > clipsquare[0]);
        default:
            return 0;
    }
}

__device__ void clip_Polygon(ScalarType* polygon, const int& length, ScalarType *clipsquare, ScalarType* newPolygon, int& newLength)
{
    ScalarType inputPolygon[max_edge_num];
    int counter = 0;
    for(int i = 0; i < 2*length; i ++)
    {
        newPolygon[i] = polygon[i];
    }
    newLength = length;

    for(int j = 0; j < 4; j++)
    {
        for(int k = 0; k < 2*newLength; k++)
        {
            inputPolygon[k] = newPolygon[k];
        }
        counter = 0;
        for(int i = 0; i < newLength; i++)
        {
            int next_id = (i + 1) % newLength;
            ScalarType s[2] = {inputPolygon[2*i],inputPolygon[2*i+1]};
            ScalarType e[2] = {inputPolygon[2*next_id],inputPolygon[2*next_id+1]};
            ScalarType intersect[2];
           
            if(inside(e, clipsquare, j))
            {
                if(inside(s, clipsquare, j))
                {
                    newPolygon[counter*2] = e[0];
                    newPolygon[counter*2+1] = e[1];
                    counter++;
                } 
                else
                {
                    intersection(s, e, clipsquare, j, intersect);
                    newPolygon[counter*2] = intersect[0];
                    newPolygon[counter*2+1] = intersect[1];
                    counter++;

                    newPolygon[counter*2] = e[0];
                    newPolygon[counter*2+1] = e[1];
                    counter++;
                } 
            }
            else if(inside(s, clipsquare, j))
            {
                intersection(s, e, clipsquare, j, intersect);
                newPolygon[counter*2] = intersect[0];
                newPolygon[counter*2+1] = intersect[1];
                counter++;
            }
        }
        newLength = counter;
    }
}

__device__ ScalarType compute_area(ScalarType* new_poly, int new_poly_num)
{
    ScalarType area=0;
    for(int i=1;i<new_poly_num-1;i++)
    {
        area+=0.5*((new_poly[2*i+1]-new_poly[1])*(new_poly[2*i+2]-new_poly[0])-(new_poly[2*i]-new_poly[0])*(new_poly[2*i+3]-new_poly[1]));
    }
    if(area<0)
    {
        area=-1*area;
    }
    return area;
}