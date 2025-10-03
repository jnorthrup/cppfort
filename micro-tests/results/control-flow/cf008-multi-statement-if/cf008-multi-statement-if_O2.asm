
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf008-multi-statement-if/cf008-multi-statement-if_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z20test_multi_statementi>:
100000360: 531f7808    	lsl	w8, w0, #1
100000364: 11002908    	add	w8, w8, #0xa
100000368: 7100001f    	cmp	w0, #0x0
10000036c: 5a9fc100    	csinv	w0, w8, wzr, gt
100000370: d65f03c0    	ret

0000000100000374 <_main>:
100000374: 52800280    	mov	w0, #0x14               ; =20
100000378: d65f03c0    	ret
