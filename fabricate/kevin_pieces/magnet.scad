include <defaults.scad>
//designed to subtract from the base at origin
echo("in Magnet")
echo(ROTATE);
echo(Half_Pieces);

module magnet(bottom_offset = DEFAULT_BOTTOM_OFFSET, magnet_dia=DEFAULT_MAGNET_DIA, magnet_h=DEFAULT_MAGNET_H, $fn=64){
    y = (Half_Pieces) ? -(magnet_dia/2 + 2): 0;
    translate([0,y,bottom_offset])cylinder(h=magnet_h,d=magnet_dia);
}

module king_magnet(bottom_offset = DEFAULT_BOTTOM_OFFSET, magnet_dia=DEFAULT_MAGNET_DIA, magnet_h=DEFAULT_MAGNET_H, $fn=64,ducky=false,
       				 o=King_Height*.7){
    y = (Half_Pieces) ? -(magnet_dia/2 + 2): 0;
    color([1,0,0])translate([0,y,bottom_offset])cylinder(h=magnet_h,d=magnet_dia);
    echo (Half_Pieces);
    if (Half_Pieces && Two_magnets){
    echo("placing second magnet");
       color([1,0,0])translate([0,-(o),bottom_offset])cylinder(h=magnet_h,d=magnet_dia);
    }
}

module tab_handle(){
if (Handles){  
    translate([-5,10,15])rotate([0,90,30])cylinder(h=2, d=15);
    translate([5,10,15])rotate([0,90,-30])cylinder(h=2, d=15);
    }
}
module weight_pocket(dia = 16.2,offset=.4){
    translate([0,0,dia/2+offset])union(){
        translate([0,0,0])sphere(d=dia);
        cylinder(d=dia,h=dia/2);
    }
}

module hex_pocket(o=.4){
    $fn=6;
    translate([0,0,o])cylinder(d=16, h=20);
}
module penny_pocket(o=.4,n=5){
    translate([0,0,o])cylinder($fn =99, d=19.5,h=1.4*n+.2);
}
/*
difference(){
    cylinder(d=18, h= 11, $fn=99);
    translate([0,0,2])hex_pocket();
}*/