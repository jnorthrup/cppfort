
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf009-if-side-effect/cf009-if-side-effect_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_if_side_effectiRi>:
100000360: 7100041f    	cmp	w0, #0x1
100000364: 5400006b    	b.lt	0x100000370 <__Z19test_if_side_effectiRi+0x10>
100000368: 531f7808    	lsl	w8, w0, #1
10000036c: b9000028    	str	w8, [x1]
100000370: d65f03c0    	ret

0000000100000374 <_main>:
100000374: 52800140    	mov	w0, #0xa                ; =10
100000378: d65f03c0    	ret
