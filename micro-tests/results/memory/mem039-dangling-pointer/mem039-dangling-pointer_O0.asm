
/Users/jim/work/cppfort/micro-tests/results/memory/mem039-dangling-pointer/mem039-dangling-pointer_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16dangling_pointerv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 910033e0    	add	x0, sp, #0xc
100000368: 52800548    	mov	w8, #0x2a               ; =42
10000036c: b9000fe8    	str	w8, [sp, #0xc]
100000370: 910043ff    	add	sp, sp, #0x10
100000374: d65f03c0    	ret

0000000100000378 <_main>:
100000378: d10043ff    	sub	sp, sp, #0x10
10000037c: 52800000    	mov	w0, #0x0                ; =0
100000380: b9000fff    	str	wzr, [sp, #0xc]
100000384: 910043ff    	add	sp, sp, #0x10
100000388: d65f03c0    	ret
