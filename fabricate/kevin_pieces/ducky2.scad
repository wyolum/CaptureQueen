include <magnet.scad>
segments = 64;

module ducky(){
    union(){
        rotate_extrude(convexity = 10, $fn = segments) {
          import(file = "profiles/my_knight_profile_0.svg");
        }
        difference(){
            translate([0,0,24])rotate([-25.0,0])scale([.25,.25,.25])import("Rubber_Duck.stl");
            cylinder(d=40,h=15);
        }
}
}

difference(){
   union(){
        translate([0,0,lift])rotate(ROTATE)scale(SCALE)ducky();
        if (lift != 0)
            translate([0, 0, 0])linear_extrude(height=lift)projection()rotate([90, 0, 0])scale(SCALE)knight();
    }
    //chop off the bottom in case we are rotated
    rotate([180,0,0])cylinder(h=King_Height,d=300);
    //magnet();
    king_magnet(o=.4*King_Height);
}
