include <defaults.scad>
//designed to subtract from the base at origin
echo("in Magnent")
echo(ROTATE);
echo(Half_Pieces);

module magnet(bottom_offset = DEFAULT_BOTTOM_OFFSET, magnet_dia=DEFAULT_MAGNET_DIA, magnet_h=DEFAULT_MAGNET_H, $fn=64){
    y = (Half_Pieces) ? -(magnet_dia/2 + 2): 0;
    translate([0,y,bottom_offset])cylinder(h=magnet_h,d=magnet_dia);
}