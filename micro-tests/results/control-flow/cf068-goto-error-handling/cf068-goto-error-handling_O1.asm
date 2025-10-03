
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf068-goto-error-handling/cf068-goto-error-handling_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15test_goto_errori>:
100000360: 12800008    	mov	w8, #-0x1               ; =-1
100000364: 12800029    	mov	w9, #-0x2               ; =-2
100000368: 531f780a    	lsl	w10, w0, #1
10000036c: 1280004b    	mov	w11, #-0x3              ; =-3
100000370: 7101901f    	cmp	w0, #0x64
100000374: 1a8ac16a    	csel	w10, w11, w10, gt
100000378: 7100001f    	cmp	w0, #0x0
10000037c: 1a8a0129    	csel	w9, w9, w10, eq
100000380: 7201001f    	tst	w0, #0x80000000
100000384: 1a891100    	csel	w0, w8, w9, ne
100000388: d65f03c0    	ret

000000010000038c <_main>:
10000038c: 52800c80    	mov	w0, #0x64               ; =100
100000390: d65f03c0    	ret
