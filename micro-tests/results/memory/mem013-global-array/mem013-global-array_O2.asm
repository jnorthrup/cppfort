
/Users/jim/work/cppfort/micro-tests/results/memory/mem013-global-array/mem013-global-array_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003f8 <__Z17test_global_arrayv>:
1000003f8: 90000028    	adrp	x8, 0x100004000 <_global_arr>
1000003fc: b9400900    	ldr	w0, [x8, #0x8]
100000400: d65f03c0    	ret

0000000100000404 <_main>:
100000404: 90000028    	adrp	x8, 0x100004000 <_global_arr>
100000408: b9400900    	ldr	w0, [x8, #0x8]
10000040c: d65f03c0    	ret
