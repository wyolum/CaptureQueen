include <magnet.scad>
segments = 64;

module pawn(){
    rotate_extrude(convexity = 10, $fn = segments) {
    import(file = "profiles/my_pawn_profile.svg");
  }
  }
 
difference(){
   rotate(ROTATE)scale(SCALE)pawn();
    //chop off the bottom in case we are rotated
    rotate([180,0,0])cylinder(h=King_Height,d=300);
    magnet();
}


