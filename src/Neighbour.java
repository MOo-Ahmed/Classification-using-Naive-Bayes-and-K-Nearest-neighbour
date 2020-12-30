import java.util.Comparator;

public class Neighbour implements Comparator<Neighbour> {
    public double distance = Integer.MAX_VALUE ;
    public String classLabel = "" ;

    public Neighbour(){

    }

    public Neighbour(double d, String c){
        setDistance(d);
        setClassLabel(c);
    }

    public void setDistance(double d){
        this.distance = d ;
    }

    public void setClassLabel(String c){
        this.classLabel = c ;
    }

    @Override
    public int compare(Neighbour o1, Neighbour o2) {
        if(o1.distance > o2.distance)   return 1 ;
        else if(o1.distance < o2.distance)   return -1 ;
        return 0;
    }

    
}