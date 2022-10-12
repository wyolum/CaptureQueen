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
}

difference(){
   rotate(ROTATE)scale(SCALE)rook();
    //chop off the bottom in case we are rotated
    rotate([180,0,0])cylinder(h=King_Height,d=300);
    //magnet();
    king_magnet(o=King_Height*.42);
}