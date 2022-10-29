include <magnet.scad>
segments = 64;

module rook(){
    difference(){
    rotate_extrude(convexity = 10, $fn = segments) {
        import(file = "profiles/my_rook_profile.svg");
    }
    color("red")translate([-50, -1.5, 46])cube([100, 3, 10]);
    rotate([0, 0, 90])color("red")translate([-50, -1.5, 46])cube([100, 3, 10]);
    }
    tab_handle();
}

difference(){
   union(){
        translate([0,0,lift])rotate(ROTATE)scale(SCALE)rook();
        if (lift != 0)
            translate([0, 0, 0])linear_extrude(height=lift)projection(cut=true)rotate([90, 0, 0])scale(SCALE)rook();
    }
    //chop off the bottom in case we are rotated
    rotate([180,0,0])cylinder(h=King_Height,d=300);
    //magnet();
    king_magnet(o=King_Height*.42);
}