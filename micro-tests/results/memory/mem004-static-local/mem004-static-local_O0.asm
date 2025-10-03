
/Users/jim/work/cppfort/micro-tests/results/memory/mem004-static-local/mem004-static-local_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003f8 <__Z17test_static_localv>:
1000003f8: 90000028    	adrp	x8, 0x100004000 <__ZZ17test_static_localvE7counter>
1000003fc: b9400109    	ldr	w9, [x8]
100000400: 11000520    	add	w0, w9, #0x1
100000404: b9000100    	str	w0, [x8]
100000408: d65f03c0    	ret

000000010000040c <_main>:
10000040c: d10083ff    	sub	sp, sp, #0x20
100000410: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000414: 910043fd    	add	x29, sp, #0x10
100000418: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000041c: 97fffff7    	bl	0x1000003f8 <__Z17test_static_localv>
100000420: 97fffff6    	bl	0x1000003f8 <__Z17test_static_localv>
100000424: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000428: 910083ff    	add	sp, sp, #0x20
10000042c: d65f03c0    	ret
