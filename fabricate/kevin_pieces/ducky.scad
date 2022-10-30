include <magnet.scad>
segments = 64;

module ducky(){
    union(){
        rotate_extrude(convexity = 10, $fn = segments) {
          import(file = "profiles/my_knight_profile_0.svg");
        }
        difference(){
            translate([0,0,24])rotate([0,0,90])scale(.3)import("Rubber_Duck.stl");
            cylinder(d=40,h=15);
        }
}
}

difference(){
   union(){
        translate([0,0,lift])rotate(ROTATE)scale(SCALE)ducky();
        if (lift != 0)
            translate([0, 0, 0])linear_extrude(height=lift)projection(cut=false)rotate([90, 0, 0])scale(SCALE)ducky();
    }
    //chop off the bottom in case we are rotated
    rotate([180,0,0])cylinder(h=King_Height,d=300);
    king_magnet(o=.3*King_Height);
    //penny_pocket();
}
