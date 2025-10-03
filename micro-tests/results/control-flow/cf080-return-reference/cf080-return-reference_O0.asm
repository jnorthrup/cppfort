
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf080-return-reference/cf080-return-reference_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003f8 <__Z21test_return_referencev>:
1000003f8: 90000020    	adrp	x0, 0x100004000 <_global>
1000003fc: 91000000    	add	x0, x0, #0x0
100000400: d65f03c0    	ret

0000000100000404 <_main>:
100000404: d10083ff    	sub	sp, sp, #0x20
100000408: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000040c: 910043fd    	add	x29, sp, #0x10
100000410: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000414: 97fffff9    	bl	0x1000003f8 <__Z21test_return_referencev>
100000418: f90003e0    	str	x0, [sp]
10000041c: f94003e9    	ldr	x9, [sp]
100000420: 52800c88    	mov	w8, #0x64               ; =100
100000424: b9000128    	str	w8, [x9]
100000428: 90000028    	adrp	x8, 0x100004000 <_global>
10000042c: b9400100    	ldr	w0, [x8]
100000430: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000434: 910083ff    	add	sp, sp, #0x20
100000438: d65f03c0    	ret
