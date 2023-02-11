$fn = 64;
scale_factor = 50/88.;
module king(){
  scale(scale_factor){
    rotate_extrude(convexity=10) {
      import(file = "profiles/my_king_profile.svg");
    }
    
    translate([-6.75, 0, 73.86])rotate([90, 0, 0])translate([0, 0, -2])linear_extrude(height=4)
      import(file="profiles/my_king_cross.svg");
  }
}

module half_king(){
  difference(){
    translate([0, 0, 1])rotate([90, 0, 0])king();
    translate([-100, -100, -200])cube(200);
  }
}

half_king();
