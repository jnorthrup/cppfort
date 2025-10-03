
/Users/jim/work/cppfort/micro-tests/results/memory/mem028-const-pointer/mem028-const-pointer_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z18test_const_pointerv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 910033e8    	add	x8, sp, #0xc
100000368: 52800549    	mov	w9, #0x2a               ; =42
10000036c: b9000fe9    	str	w9, [sp, #0xc]
100000370: f90003e8    	str	x8, [sp]
100000374: f94003e8    	ldr	x8, [sp]
100000378: b9400100    	ldr	w0, [x8]
10000037c: 910043ff    	add	sp, sp, #0x10
100000380: d65f03c0    	ret

0000000100000384 <_main>:
100000384: d10083ff    	sub	sp, sp, #0x20
100000388: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000038c: 910043fd    	add	x29, sp, #0x10
100000390: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000394: 97fffff3    	bl	0x100000360 <__Z18test_const_pointerv>
100000398: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000039c: 910083ff    	add	sp, sp, #0x20
1000003a0: d65f03c0    	ret
