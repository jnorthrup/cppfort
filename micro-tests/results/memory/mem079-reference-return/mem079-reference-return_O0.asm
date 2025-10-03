
/Users/jim/work/cppfort/micro-tests/results/memory/mem079-reference-return/mem079-reference-return_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003f8 <__Z13get_referencev>:
1000003f8: 90000020    	adrp	x0, 0x100004000 <__ZZ13get_referencevE1x>
1000003fc: 91000000    	add	x0, x0, #0x0
100000400: d65f03c0    	ret

0000000100000404 <_main>:
100000404: d10083ff    	sub	sp, sp, #0x20
100000408: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000040c: 910043fd    	add	x29, sp, #0x10
100000410: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000414: 97fffff9    	bl	0x1000003f8 <__Z13get_referencev>
100000418: f90003e0    	str	x0, [sp]
10000041c: f94003e8    	ldr	x8, [sp]
100000420: b9400100    	ldr	w0, [x8]
100000424: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000428: 910083ff    	add	sp, sp, #0x20
10000042c: d65f03c0    	ret
