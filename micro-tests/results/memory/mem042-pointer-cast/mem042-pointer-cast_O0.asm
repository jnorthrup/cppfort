
/Users/jim/work/cppfort/micro-tests/results/memory/mem042-pointer-cast/mem042-pointer-cast_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_pointer_castv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 910033e8    	add	x8, sp, #0xc
100000368: 528acf09    	mov	w9, #0x5678             ; =22136
10000036c: 72a24689    	movk	w9, #0x1234, lsl #16
100000370: b9000fe9    	str	w9, [sp, #0xc]
100000374: f90003e8    	str	x8, [sp]
100000378: f94003e8    	ldr	x8, [sp]
10000037c: 39c00100    	ldrsb	w0, [x8]
100000380: 910043ff    	add	sp, sp, #0x10
100000384: d65f03c0    	ret

0000000100000388 <_main>:
100000388: d10083ff    	sub	sp, sp, #0x20
10000038c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000390: 910043fd    	add	x29, sp, #0x10
100000394: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000398: 97fffff2    	bl	0x100000360 <__Z17test_pointer_castv>
10000039c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a0: 910083ff    	add	sp, sp, #0x20
1000003a4: d65f03c0    	ret
