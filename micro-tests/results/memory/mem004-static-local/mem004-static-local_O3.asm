
/Users/jim/work/cppfort/micro-tests/results/memory/mem004-static-local/mem004-static-local_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003f8 <__Z17test_static_localv>:
1000003f8: 90000028    	adrp	x8, 0x100004000 <__ZZ17test_static_localvE7counter>
1000003fc: b9400109    	ldr	w9, [x8]
100000400: 11000520    	add	w0, w9, #0x1
100000404: b9000100    	str	w0, [x8]
100000408: d65f03c0    	ret

000000010000040c <_main>:
10000040c: 90000028    	adrp	x8, 0x100004000 <__ZZ17test_static_localvE7counter>
100000410: b9400109    	ldr	w9, [x8]
100000414: 11000920    	add	w0, w9, #0x2
100000418: b9000100    	str	w0, [x8]
10000041c: d65f03c0    	ret
