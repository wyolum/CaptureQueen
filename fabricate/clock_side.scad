height = 64;
top_width = 40;
base_width = 80;

thickness = 10;

nudge = 1;
module start_side(){
  linear_extrude(height=10)polygon(points=[
					   [0, 0],
					   [0, height],
					   [top_width, height],
					   [base_width, height/2],
					   [base_width, 0],

					   [thickness, thickness],
					   [thickness, height - thickness],
					   [top_width - nudge, height - thickness],
					   [base_width - thickness, height/2 - nudge],
					   [base_width - thickness, thickness],

					   ],
				   paths = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]);
}

module screw(){
  translate([0, 10, 6.5])rotate([90, 0, 0])cylinder(h=20, d=2, $fn=20);
}

module holes(){
  // bottom
  translate([10, 0, 0])screw();
  translate([70, 0, 0])screw();

  //front
  translate([80, 10, 0])rotate([0, 0, 90])screw();
  translate([80, 25, 0])rotate([0, 0, 90])screw();

  //diag
  rotate([0, 0, 145])translate([-38, -75, 0])screw();
  rotate([0, 0, 145])translate([-3, -75, 0])screw();

  // back
  translate([0, 10, 0])rotate([0, 0, -90])screw();
  translate([0, height-10, 0])rotate([0, 0, -90])screw();

  // top
  #translate([10, height, 0])rotate([0, 0, 180])screw();
  #translate([top_width-10, height, 0])rotate([0, 0, 180])screw();
}

difference(){
  start_side();
  translate([thickness/2, thickness/2, thickness/4])scale([.85, .85, 2])start_side();
  holes();
}

