
/Users/jim/work/cppfort/micro-tests/results/memory/mem029-pointer-const/mem029-pointer-const_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z18test_pointer_constv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 910033e8    	add	x8, sp, #0xc
100000368: 52800549    	mov	w9, #0x2a               ; =42
10000036c: b9000fe9    	str	w9, [sp, #0xc]
100000370: 52800149    	mov	w9, #0xa                ; =10
100000374: b9000be9    	str	w9, [sp, #0x8]
100000378: f90003e8    	str	x8, [sp]
10000037c: f94003e8    	ldr	x8, [sp]
100000380: b9400100    	ldr	w0, [x8]
100000384: 910043ff    	add	sp, sp, #0x10
100000388: d65f03c0    	ret

000000010000038c <_main>:
10000038c: d10083ff    	sub	sp, sp, #0x20
100000390: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000394: 910043fd    	add	x29, sp, #0x10
100000398: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000039c: 97fffff1    	bl	0x100000360 <__Z18test_pointer_constv>
1000003a0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a4: 910083ff    	add	sp, sp, #0x20
1000003a8: d65f03c0    	ret
