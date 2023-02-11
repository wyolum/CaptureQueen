$fn = 64;
scale_factor = 50/88;

module bishop(){
  rotate([0, 0, 180])scale(scale_factor)
  difference(){
    scale([.9, .9, 1])rotate_extrude(convexity = 10) {
      import(file = "profiles/my_bishop_profile.svg");
    }
    color("red")translate([-50, -3, 48])rotate([25, 0, 0])cube([100, 2, 10]);
  }
}

module half_bishop(){
  difference(){
    translate([0, 0, 1])rotate([90, 0, 0])bishop();
    translate([-50, -50, -100])cube(100);
  }
}

half_bishop();

