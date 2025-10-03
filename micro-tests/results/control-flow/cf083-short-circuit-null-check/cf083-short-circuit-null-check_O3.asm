
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf083-short-circuit-null-check/cf083-short-circuit-null-check_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15test_null_checkPi>:
100000360: b40000a0    	cbz	x0, 0x100000374 <__Z15test_null_checkPi+0x14>
100000364: b9400000    	ldr	w0, [x0]
100000368: 7100001f    	cmp	w0, #0x0
10000036c: 5400004d    	b.le	0x100000374 <__Z15test_null_checkPi+0x14>
100000370: d65f03c0    	ret
100000374: 52800000    	mov	w0, #0x0                ; =0
100000378: d65f03c0    	ret

000000010000037c <_main>:
10000037c: 52800540    	mov	w0, #0x2a               ; =42
100000380: d65f03c0    	ret
