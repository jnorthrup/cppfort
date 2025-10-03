
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf077-return-in-loop/cf077-return-in-loop_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_return_in_loopi>:
100000360: aa0003e8    	mov	x8, x0
100000364: 52800c69    	mov	w9, #0x63               ; =99
100000368: 71018c1f    	cmp	w0, #0x63
10000036c: 1a893000    	csel	w0, w0, w9, lo
100000370: 52800c89    	mov	w9, #0x64               ; =100
100000374: 340000a8    	cbz	w8, 0x100000388 <__Z19test_return_in_loopi+0x28>
100000378: 51000508    	sub	w8, w8, #0x1
10000037c: 71000529    	subs	w9, w9, #0x1
100000380: 54ffffa1    	b.ne	0x100000374 <__Z19test_return_in_loopi+0x14>
100000384: 12800000    	mov	w0, #-0x1               ; =-1
100000388: d65f03c0    	ret

000000010000038c <_main>:
10000038c: 52800540    	mov	w0, #0x2a               ; =42
100000390: d65f03c0    	ret
