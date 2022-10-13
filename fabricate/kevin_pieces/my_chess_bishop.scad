include <magnet.scad>
segments = 64;

module bishop(){
    rotate([0,0,180])difference(){
      scale([.9, .9, 1])rotate_extrude(convexity = 10, $fn = segments) {
        import(file = "profiles/my_bishop_profile.svg");
      }
      color("red")translate([-50, -3, 48])rotate([25, 0, 0])cube([100, 2, 10]);
    }
    color("blue")translate([0,0,48])sphere(d=10, $fn=99);
    translate([0,-3,-5])tab_handle();
}
difference(){
   rotate(ROTATE)scale(SCALE)bishop();
    //chop off the bottom in case we are rotated
    rotate([180,0,0])cylinder(h=King_Height,d=300);
    //magnet();
    king_magnet(o=King_Height *.52);
}

