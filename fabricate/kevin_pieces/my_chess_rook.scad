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
    rook();
    magnet();
}