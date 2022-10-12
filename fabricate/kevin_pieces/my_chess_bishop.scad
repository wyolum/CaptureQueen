include <magnet.scad>
segments = 64;

module bishop(){
    difference(){
      scale([.9, .9, 1])rotate_extrude(convexity = 10, $fn = segments) {
        import(file = "profiles/my_bishop_profile.svg");
      }
      color("red")translate([-50, -3, 48])rotate([25, 0, 0])cube([100, 2, 10]);
    }
}
difference(){
   rotate(ROTATE)scale(SCALE)bishop();
    //chop off the bottom in case we are rotated
    rotate([180,0,0])cylinder(h=King_Height,d=300);
    magnet();
}

