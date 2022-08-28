inch = 25.4;
OD = 24.5; // od of lager tube
snug_id_small = 19.7;
insert_diameter = 24.25;

module tube(od, id, h){
  difference(){
    color([1, .3, .6])cylinder(d=od, h=h);
    translate([0, 0, -1])cylinder(d=id, h=h + 2);
  }
}

