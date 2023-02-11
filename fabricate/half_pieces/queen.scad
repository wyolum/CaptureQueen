$fn=64;
scale_factor = 50/88;

module queen(){
  scale(scale_factor);
  difference(){
    rotate_extrude(convexity=10) {
      import(file = "profiles/my_queen_profile.svg");
    }
    for(i=[1:8]){
      rotate([0, 0, i * 360/8])color("black")translate([0, 52, 0])rotate([30, 0, 0])scale([1, .25, 1])cylinder(d=12, h=200, $fn=30);
    }
  }
}

module half_queen(){
  difference(){
    translate([0, 0, 1])rotate([90, 0, 0])queen();
    translate([-100, -100, -200])cube(200);
  }
}

half_queen();

