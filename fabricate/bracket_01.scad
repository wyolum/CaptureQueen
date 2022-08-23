module tube(od, id, h){
  difference(){
    cylinder(d=od, h=h);
    translate([0, 0, -1])cylinder(d=id, h=h + 2);
  }
}

OD = 20;
ID = 16;
//t=20;
//rotate([22.5 * sin(t * 360) - 45 + 22.5, 0, 0])
translate([0, OD ,-50]){
  difference(){
    rotate([45, 0, 0])translate([0, OD, 0])tube(OD, ID, 300);
    cylinder(h=100, d=OD + 4);
    translate([-50, 0, 0])cube(100);
  }
}

translate([0, OD+2, -30])tube(OD+4, ID+4, 60);

translate([-OD/2-6, OD + 4, -2 * OD])rotate([90+45, 0, 0])rotate([0, 90, 0])linear_extrude(height=6)scale(3* OD)polygon(points=[[0, 0], [1, 0], [0, 1]]);
translate([OD/2, OD + 4, -2 * OD])rotate([90+45, 0, 0])rotate([0, 90, 0])linear_extrude(height=6)scale(3* OD)polygon(points=[[0, 0], [1, 0], [0, 1]]);
