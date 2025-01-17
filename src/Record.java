public class Record {
    String BuyingPrice = "" ;
    String CarID = "" ;
    String MaintenancePrice = "" ;
    String NumberOfDoors = "" ;
    String Capacity = "" ;
    String SizeOfLuggageBoot = "" ;
    String EstimatedSafety = "" ;
    String CarAcceptability = "" ;

    public Record(){}

    public Record(String carid, String bPrice, String mPrice, String doors, String cap, String luggage, String safety){
        this.CarID = carid ;
        this.BuyingPrice = bPrice ;
        this.MaintenancePrice = mPrice ;
        this.NumberOfDoors = doors ;
        this.Capacity = cap ;
        this.SizeOfLuggageBoot = luggage ;
        this.EstimatedSafety = safety ;
    }

    public void setCarAcceptability (String acc){
        this.CarAcceptability = acc ;
    }

    @Override
    public String toString(){
        return CarID + "," + BuyingPrice + "," + MaintenancePrice  + "," + NumberOfDoors  + "," 
        + Capacity + "," + SizeOfLuggageBoot + "," + EstimatedSafety  + "," + CarAcceptability + "\n" ;
    }
}
