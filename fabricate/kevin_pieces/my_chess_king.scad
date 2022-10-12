include <magnet.scad>
segments = 64;

module king(){
  rotate_extrude(convexity = 10, $fn = segments) {
    import(file = "profiles/my_king_profile.svg");
  }

  translate([-6.75, 0, 73.86])rotate([90, 0, 0])translate([0, 0, -2])linear_extrude(height=4)
    import(file="profiles/my_king_cross.svg");
}
difference(){
   rotate(ROTATE)scale(SCALE)king();
    //chop off the bottom in case we are rotated
    rotate([180,0,0])cylinder(h=King_Height,d=300);
    //magnet();
    king_magnet();
}
