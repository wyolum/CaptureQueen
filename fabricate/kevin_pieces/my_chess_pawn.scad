include <magnet.scad>
segments = 64;

module pawn(){
    rotate_extrude(convexity = 10, $fn = segments) {
    import(file = "profiles/my_pawn_profile.svg");
  }
  translate([0,-5,-5])tab_handle();
  }
 
difference(){
   union(){
        translate([0,0,lift])rotate(ROTATE)scale(SCALE)pawn();
        if (lift != 0)
            translate([0, 0, 0])linear_extrude(height=lift)projection()rotate([90, 0, 0])scale(SCALE)pawn();
    }
    //chop off the bottom in case we are rotated
    rotate([180,0,0])cylinder(h=King_Height,d=300);
    magnet();
}


