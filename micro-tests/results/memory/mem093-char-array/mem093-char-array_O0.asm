
/Users/jim/work/cppfort/micro-tests/results/memory/mem093-char-array/mem093-char-array_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003b0 <__Z15test_char_arrayv>:
1000003b0: d10043ff    	sub	sp, sp, #0x10
1000003b4: 90000008    	adrp	x8, 0x100000000
1000003b8: 910fe108    	add	x8, x8, #0x3f8
1000003bc: b9400109    	ldr	w9, [x8]
1000003c0: b9000be9    	str	w9, [sp, #0x8]
1000003c4: 79400908    	ldrh	w8, [x8, #0x4]
1000003c8: 79001be8    	strh	w8, [sp, #0xc]
1000003cc: 39c023e0    	ldrsb	w0, [sp, #0x8]
1000003d0: 910043ff    	add	sp, sp, #0x10
1000003d4: d65f03c0    	ret

00000001000003d8 <_main>:
1000003d8: d10083ff    	sub	sp, sp, #0x20
1000003dc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003e0: 910043fd    	add	x29, sp, #0x10
1000003e4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003e8: 97fffff2    	bl	0x1000003b0 <__Z15test_char_arrayv>
1000003ec: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003f0: 910083ff    	add	sp, sp, #0x20
1000003f4: d65f03c0    	ret
