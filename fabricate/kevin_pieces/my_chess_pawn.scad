include <magnet.scad>
segments = 64;

module pawn(){
    rotate_extrude(convexity = 10, $fn = segments) {
    import(file = "profiles/my_pawn_profile.svg");
  }
  }
 
difference(){
    pawn();
    magnet();
}


