(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* The property of states by which the check of regions
   (FlambdaRegionsCheckSpec.v) is to be proved sound.

   This file is to be reviewed: all of it. It adds nothing to the
   machine and nothing to the check. Nothing proves the statements at
   the end yet: the file is here so that the property is read, and
   broken if it can be, before the proof that a step keeps it is
   written.

   THE PLAN OF THE PROOF

   [state_checked] is a property of a state of the machine of regions,
   given a certificate and the code of the term. The state that a run
   of a checked term starts in has it ([checked_initial_statement]) and
   a step keeps it ([region_step_keeps_checked_statement]). A state
   that has it stops without a step only where [step] has none
   ([checked_fault_statement]): not for decisions D45, D46 or D47, not
   for a region used as a value, not for a primitive of regions that
   is not as the conversion makes it. And where [step] has none, the
   run of the term with its regions dropped goes wrong
   ([no_plain_step_erased_statement]); that part needs of the states
   of the run only [reads_live], which a checked state has
   ([checked_reads_live_statement]). The four give
   FlambdaRegionsCheckSpec.region_check_sound_statement.

   SYMBOLS AS TIMES

   The machine draws its symbols upwards, so a symbol says when it was
   drawn, and the frames are in the order of their marks
   (FlambdaRegionsCallsSpec.frames_ordered): a frame binds symbols
   drawn after its mark and before the mark of the frame above it. So
   "in a frame below the frame of the mark [m]" is "in a frame, and
   drawn before [m]", and the end of a region whose mark is [m] frees
   nothing that was drawn before [m].

   A value is within a bound ([value_within]) if every symbol that it
   reaches without reading the static environment is of the heap, or
   is in a frame and was drawn before the bound. "Reaches" goes into
   a partial application, unlike
   FlambdaRegionsInvariantSpec.value_holds.

   WHAT A LIFETIME SAYS OF A VALUE ([lifetime_holds])

   Given the open regions of the check, newest first, and the marks
   that their variables are bound to:
   - [Forever]: every symbol that the value reaches is of the heap;
   - [Of_region r]: if the mark of [r] is on the stack, the value is
     within the mark of the next open region above [r]; within no
     bound if [r] is the newest;
   - [Of_caller]: the value is within the mark of the oldest open
     region; within no bound if no region is open;
   - [Gone], or a region that is not open: nothing.
   The bound of the newest region, and of the code where no region is
   open, is a parameter: none for the state that runs, and the counter
   of the machine for a state that is kept (below).

   THE STATE THAT RUNS ([running_checked])

   There is a state of the check under which the expression of the
   state passes; its open regions are bound to marks that are on the
   stack and that decrease; each variable holds a value as its
   lifetime says.

   THE CONTINUATIONS ([cont_ok])

   The machine has no stack of calls: the callers are what the
   continuations hold. Each continuation that the state of the check
   has an entry for stands, in the state, for a value that is as the
   entry says, to a user: the variables, the variables of regions, the
   marks of the open regions and the counter of whoever may apply it.

   A handler ([Ok_handler]) is a closure. The clause is of the state
   of the check where it was made, with the marks of its open regions
   and the counter then ([made_at]):
   - the handler, and those of its recursive group, pass the check
     there;
   - the variables that the closure holds are as their lifetimes say,
     with the marks as they were and the counter then as the bound of
     the newest region: a value that the closure holds was drawn
     before it was made;
   - every mark of the user that was drawn before the closure was made
     is the mark of a region that was open where it was made. So the
     end of a region by the user frees nothing that the closure counts
     on: the region is one that the closure knows, and then it is at
     or below the region whose values are freed, or above the bound
     of what is kept; or it was begun after the closure was made, and
     is above everything that the closure holds;
   - the variables of regions that were in scope where the closure was
     made are bound, for the user, to what they were bound to: no
     variable of a region is bound twice;
   - the continuations that the closure holds are as their entries
     say, to the closure as their user.

   The return continuation of code ([Ok_return]) stands for what the
   caller gave. The clause is of the caller as it was at the call
   ([caller]): the state of the check, the variables, the marks of the
   open regions and the counter. The marks of the caller are still on
   the stack, and every mark of the user was drawn after the call
   ([called_from]): code is given no region, so it can end none of its
   caller's. What the caller gave takes one value of a lifetime of the
   caller ([ret_ok]): it is a continuation of the caller that the
   check let the result go to ([Ret_to]); or what is left of an
   application with too many arguments, whose arguments are as their
   lifetimes say and are the arguments of a call that the check allows
   ([Ret_over]). The exception continuation ([Ok_raise], [raise_ok])
   is a continuation of the caller, with the extra arguments that the
   call gave it.

   A continuation with no entry is unreachable, and nothing is asked
   of what it stands for; one that the state of the check does not
   bind is a way out of the term, and the state binds it to nothing.

   THE REST

   - The code of the static environment is the code that the check was
     given, and each passes the check ([code_checked]).
   - A closure that a frame binds is of code whose closures may be
     local ([local_closures_fit]).
   - What a frame binds reaches symbols of the heap, of that frame or
     of a frame below it, and what the static environment binds
     symbols of the heap, a partial application included
     ([frames_deep], [statics_deep]).
     FlambdaRegionsInvariantSpec.frames_hold stops at a partial
     application.
   - Every partial application that the state holds, anywhere, is one
     that may be called ([paps_fit]): it keeps fewer arguments than its
     code takes, as one that a call makes does; and it reaches symbols
     of the heap only, or the certificate says that a partial
     application may keep a local value, and each argument that it
     keeps is of the heap or is at a parameter that may be given a
     local value. The first state holds none
     (FlambdaRegionsCheckSpec.statics_no_partial).

   WHAT IS NOT HERE

   - Nothing of the heap of the state: the machine of regions runs no
     primitive of the heap.
   - Nothing of the values at an exit beyond what
     FlambdaRegionsCallsSpec.v states.
   - That a region is ended (decision D48). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaRegionsSpec
  FlambdaRegionsInvariantSpec FlambdaRegionsErasureSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec.
Import ListNotations.

(* ======================================================================== *)
(* What a value reaches                                                     *)
(* ======================================================================== *)

(* Every symbol that a value reaches without reading the static
   environment is one of [ok]: itself, its fields, its elements, its
   value slots, the callee and the arguments of a partial application,
   however deep. *)
Fixpoint value_all (ok : symbol -> bool) (v : value) : bool :=
  match v with
  | Value_const _ | Value_boxed_number _ | Value_location _ => true
  | Value_closure _ value_slots | Value_closure_of_set _ _ value_slots =>
    forallb (fun '(_, v) => value_all ok v) value_slots
  | Value_symbol s => ok s
  | Value_block _ fields => forallb (value_all ok) fields
  | Value_immutable_array elements => forallb (value_all ok) elements
  | Value_partial_application callee args _ =>
    value_all ok callee && forallb (value_all ok) args
  end.

(* The same, of what a symbol is bound to. *)
Definition allocated_all (ok : symbol -> bool)
    (allocated : statically_allocated_value) : bool :=
  match allocated with
  | Static_closure _ value_slots =>
    forallb (fun '(_, v) => value_all ok v) value_slots
  | Static_closure_of_set _ closure_symbols _ value_slots =>
    forallb (fun '(_, s) => ok s) closure_symbols
    && forallb (fun '(_, v) => value_all ok v) value_slots
  | Static_block _ fields => forallb (value_all ok) fields
  | Static_boxed_number contents => value_all ok contents
  | Static_string _ | Static_imported _ => true
  | Static_immutable_array elements => forallb (value_all ok) elements
  end.

(* A symbol that may be evaluated: of the heap, or bound in a frame. *)
Definition live_symbol (rs : region_state) (s : symbol) : bool :=
  heap_symbol rs s || local_in (rs_frames rs) s.

(* Drawn before the bound; with no bound, any symbol. *)
Definition before (stop : option symbol) (s : symbol) : bool :=
  match stop with
  | Some m => Pos.ltb s m
  | None => true
  end.

(* A symbol within a bound: of the heap, or bound in a frame and drawn
   before the bound. *)
Definition symbol_within (rs : region_state) (stop : option symbol)
    (s : symbol) : bool :=
  heap_symbol rs s || (local_in (rs_frames rs) s && before stop s).

Definition value_within (rs : region_state) (stop : option symbol)
    (v : value) : bool :=
  value_all (symbol_within rs stop) v.

(* ======================================================================== *)
(* The open regions and their marks                                         *)
(* ======================================================================== *)

(* The marks of the open regions, newest first: what their variables
   are bound to. Nothing if one is not bound to a symbol. *)
Fixpoint open_marks (vars : env) (opens : list variable)
    : option (list mark) :=
  match opens with
  | [] => Some []
  | r :: opens =>
    match lookup r vars with
    | Some (Value_symbol m) =>
      let* ms := open_marks vars opens in
      Some (m :: ms)
    | _ => None
    end
  end.

(* The marks decrease: each was drawn after those that follow it. *)
Fixpoint marks_decrease (ms : list mark) : bool :=
  match ms with
  | [] => true
  | m :: ms' =>
    match ms' with
    | [] => true
    | m' :: _ => Pos.ltb m' m
    end
    && marks_decrease ms'
  end.

(* The marks are on the stack, and decrease. *)
Definition marks_placed (frames : list frame) (ms : list mark) : bool :=
  forallb (fun m => on_stack m frames) ms && marks_decrease ms.

(* The mark of the open region [r], and its bound: the mark of the open
   region above it, [above] for the newest. Nothing if [r] is not
   open. *)
Fixpoint region_place (opens : list variable) (ms : list mark)
    (above : option symbol) (r : variable) : option (mark * option symbol) :=
  match opens, ms with
  | x :: opens, m :: ms =>
    if Pos.eqb x r then Some (m, above) else region_place opens ms (Some m) r
  | _, _ => None
  end.

(* The mark of the oldest open region. *)
Fixpoint lowest (ms : list mark) : option mark :=
  match ms with
  | [] => None
  | [m] => Some m
  | _ :: ms => lowest ms
  end.

(* The bound of what lives as long as the code: the mark of its oldest
   open region, [above] if it has none. *)
Definition caller_bound (ms : list mark) (above : option symbol)
    : option symbol :=
  match lowest ms with
  | Some m => Some m
  | None => above
  end.

(* What a lifetime says of a value, with the open regions [opens] at
   the marks [ms], and [above] the bound of the newest. A region whose
   mark is no longer on the stack says nothing: what lived as long as
   it is gone. *)
Definition lifetime_holds (rs : region_state) (opens : list variable)
    (ms : list mark) (above : option symbol) (l : lifetime) (v : value)
    : Prop :=
  match l with
  | Forever => value_all (heap_symbol rs) v = true
  | Of_caller => value_within rs (caller_bound ms above) v = true
  | Of_region r =>
    match region_place opens ms above r with
    | Some (m, stop) =>
      on_stack m (rs_frames rs) = true -> value_within rs stop v = true
    | None => True
    end
  | Gone => True
  end.

(* [l] lasts at least as long as [l']: of the two, [l'] ends first. *)
Definition lifetime_eqb (l l' : lifetime) : bool :=
  match l, l' with
  | Forever, Forever | Of_caller, Of_caller | Gone, Gone => true
  | Of_region r, Of_region r' => Pos.eqb r r'
  | _, _ => false
  end.

Definition lasts (open : list variable) (l l' : lifetime) : bool :=
  lifetime_eqb (shorter open l l') l'.

(* ======================================================================== *)
(* The state that runs                                                      *)
(* ======================================================================== *)

(* The state of the check describes the state that runs: its open
   regions are variables of regions, bound to marks that are on the
   stack and decrease, and each variable holds a value as its lifetime
   says. *)
Definition running_checked (rs : region_state) (cs : check_state)
    (ms : list mark) : Prop :=
  open_marks (state_vars (rs_state rs)) (cs_open cs) = Some ms
  ∧ marks_placed (rs_frames rs) ms = true
  ∧ all_among (cs_open cs) (cs_regions cs) = true
  ∧ ∀ (x : variable) (l : lifetime) (v : value),
      lookup x (cs_env cs) = Some l ->
      lookup x (state_vars (rs_state rs)) = Some v ->
      lifetime_holds rs (cs_open cs) ms None l v.

(* ======================================================================== *)
(* Who may apply a continuation, and who called                             *)
(* ======================================================================== *)

(* What a continuation is as its entry says to: the variables, the
   variables of regions, the marks of the open regions and the counter
   of the state that may apply it. *)
Record user : Type := User {
  u_vars : env;
  u_regions : list variable;
  u_marks : list mark;
  u_limit : symbol
}.

Definition running_user (rs : region_state) (cs : check_state)
    (ms : list mark) : user :=
  User (state_vars (rs_state rs)) (cs_regions cs) ms (rs_next rs).

(* A caller, as it was at the call: the state of the check, the
   variables, the marks of the open regions and the counter. *)
Record caller : Type := Caller {
  caller_check : check_state;
  caller_vars : env;
  caller_marks : list mark;
  caller_next : symbol
}.

Definition caller_user (a : caller) : user :=
  User (caller_vars a) (cs_regions (caller_check a)) (caller_marks a)
    (caller_next a).

(* The caller of the code that the user runs. Its open regions were
   bound to its marks, which are still on the stack, and were drawn
   before the call; every mark of the user was drawn after it. *)
Definition called_from (rs : region_state) (u : user) (a : caller) : Prop :=
  open_marks (caller_vars a) (cs_open (caller_check a))
  = Some (caller_marks a)
  ∧ marks_placed (rs_frames rs) (caller_marks a) = true
  ∧ all_among (cs_open (caller_check a)) (cs_regions (caller_check a)) = true
  ∧ forallb (fun m => Pos.ltb m (caller_next a)) (caller_marks a) = true
  ∧ Pos.leb (caller_next a) (u_limit u) = true
  ∧ forallb (fun m => Pos.leb (caller_next a) m) (u_marks u) = true.

(* A value that the caller gave at the call and that a continuation
   holds since is as its lifetime says, in the caller, and was drawn
   before the call. *)
Definition caller_holds (rs : region_state) (a : caller) (l : lifetime)
    (v : value) : Prop :=
  lifetime_usable (caller_check a) l = true
  ∧ lifetime_holds rs (cs_open (caller_check a)) (caller_marks a)
      (Some (caller_next a)) l v.

(* ======================================================================== *)
(* The continuations                                                        *)
(* ======================================================================== *)

Section Checked.

Variable cert : certificate.

Variable codes : list (code_id * code).

(* A closure of handlers was made in the state [cs] of the check,
   whose open regions were at the marks [ms], when the counter was
   [next]; [base] are the entries of the continuations that the
   closure holds, and [recursive] its recursive group, whose entries
   are in front of them. See the header for each clause. *)
Definition made_at (rs : region_state) (u : user) (vars : env)
    (recursive : list (continuation * continuation_handler))
    (cs : check_state) (base : list (continuation * cont_entry))
    (ms : list mark) (next : symbol) : Prop :=
  cs_conts cs
  = map (fun '(k, h) => (k, cont_entry_of cert k h)) recursive ++ base
  ∧ forallb (fun '(k, h) => check_handler cert codes cs k h) recursive = true
  ∧ (∀ (k : continuation) (h : continuation_handler),
       In (k, h) recursive -> incl (code_of_handler h) codes)
  ∧ all_among (cs_open cs) (cs_regions cs) = true
  ∧ open_marks vars (cs_open cs) = Some ms
  ∧ marks_decrease ms = true
  ∧ forallb (fun m => Pos.ltb m next) ms = true
  ∧ Pos.leb next (u_limit u) = true
  (* What the closure holds. *)
  ∧ (∀ (x : variable) (l : lifetime) (v : value),
       lookup x (cs_env cs) = Some l ->
       lookup x vars = Some v ->
       lifetime_holds rs (cs_open cs) ms (Some next) l v)
  (* The marks of the user that were drawn before the closure was
     made. *)
  ∧ (∀ m : mark, In m (u_marks u) -> Pos.ltb m next = true -> In m ms)
  (* The variables of regions. *)
  ∧ (∀ r : variable,
       among r (cs_regions cs) = true ->
       lookup r (u_vars u) = lookup r vars ∧ among r (u_regions u) = true).

(* A continuation is as its entry says, to a user ([cont_ok]); what a
   caller gave as the return continuation takes one value of a
   lifetime of the caller ([ret_ok]); what it gave as the exception
   continuation takes the exception ([raise_ok]). *)
Inductive cont_ok (rs : region_state)
    : user -> cont_entry -> cont_value -> Prop :=
  | Ok_handler (u : user) (params : list variable)
      (metadata : handler_metadata) (handler : expr) (entry : handler_entry)
      (vars : env) (conts : cont_env)
      (recursive : list (continuation * continuation_handler))
      (cs : check_state) (base : list (continuation * cont_entry))
      (ms : list mark) (next : symbol) :
    made_at rs u vars recursive cs base ms next ->
    entry_fits cs params (hm_params_kinds metadata) entry = true ->
    check_expr cert codes
      (handler_state cs params (hm_params_kinds metadata) entry) handler
    = true ->
    incl (code_of_expr handler) codes ->
    (∀ (k : continuation) (ce : cont_entry),
       lookup k base = Some ce ->
       cont_ok rs (User vars (cs_regions cs) ms next) ce (resolve k conts)) ->
    (∀ k : continuation, lookup k base = None -> lookup k conts = None) ->
    cont_ok rs u (To_handler params (hm_params_kinds metadata) entry)
      (Closure (Handler params metadata handler) vars conts recursive)
  | Ok_unreachable (u : user) (c : cont_value) : cont_ok rs u Unreachable c
  | Ok_return (u : user) (may_be_local : bool) (a : caller) (c : cont_value) :
    called_from rs u a ->
    ret_ok rs a
      (if may_be_local then newest_lifetime (caller_check a) else Forever) c ->
    cont_ok rs u (To_return may_be_local) c
  | Ok_raise (u : user) (a : caller) (c : cont_value) :
    called_from rs u a ->
    raise_ok rs a c ->
    cont_ok rs u To_raise c
with ret_ok (rs : region_state) : caller -> lifetime -> cont_value -> Prop :=
  (* A continuation of the caller that the check let a result of the
     lifetime [l] go to. *)
  | Ret_to (a : caller) (l : lifetime) (k : continuation) (c : cont_value) :
    jump_result (caller_check a) k l = true ->
    (∀ ce : cont_entry,
       lookup k (cs_conts (caller_check a)) = Some ce ->
       cont_ok rs (caller_user a) ce c) ->
    (lookup k (cs_conts (caller_check a)) = None -> c = Outer k) ->
    ret_ok rs a l c
  (* What is left of an application with too many arguments: the value
     that it is given is called with the arguments that remain. *)
  | Ret_over (a : caller) (l l' : lifetime) (ls : list lifetime)
      (remaining : list value) (remaining_arity return_arity : arity)
      (return_to exn_to : cont_value) :
    Forall2 (caller_holds rs a) ls remaining ->
    unknown_call cert codes (caller_check a) l ls (length remaining_arity)
    = Some l' ->
    ret_ok rs a l' return_to ->
    raise_ok rs a exn_to ->
    ret_ok rs a l
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
  (* What takes a value of a lifetime takes one that lasts longer. *)
  | Ret_longer (a : caller) (l l' : lifetime) (c : cont_value) :
    lasts (cs_open (caller_check a)) l l' = true ->
    ret_ok rs a l' c ->
    ret_ok rs a l c
with raise_ok (rs : region_state) : caller -> cont_value -> Prop :=
  (* A continuation of the caller, with the values of the extra
     arguments of the call, that the check let the exception and the
     extra arguments go to. *)
  | Raise_to (a : caller) (k : continuation) (simples : list simple)
      (extra : list value) (c : cont_value) :
    Forall2
      (fun s v => caller_holds rs a (lifetime_of (caller_check a) s) v)
      simples extra ->
    jump (caller_check a) k
      ((None, Forever)
       :: map (fun s => (Some s, lifetime_of (caller_check a) s)) simples)
    = true ->
    (∀ ce : cont_entry,
       lookup k (cs_conts (caller_check a)) = Some ce ->
       cont_ok rs (caller_user a) ce c) ->
    (lookup k (cs_conts (caller_check a)) = None -> c = Outer k) ->
    raise_ok rs a (with_extra_args c extra).

(* The continuations of a state are as the entries say, and a
   continuation with no entry is bound to nothing. *)
Definition conts_checked (rs : region_state) (u : user)
    (entries : list (continuation * cont_entry)) (conts : cont_env) : Prop :=
  (∀ (k : continuation) (ce : cont_entry),
     lookup k entries = Some ce -> cont_ok rs u ce (resolve k conts))
  ∧ (∀ k : continuation, lookup k entries = None -> lookup k conts = None).

(* ======================================================================== *)
(* The code, the closures and the partial applications                      *)
(* ======================================================================== *)

(* The code of the static environment of the state is the code that the
   check was given, in which no code is bound twice, and each passes
   the check. The code that the expression of a state binds is asked
   to be in that list too ([state_checked]); of the body of code that
   a call enters it is so because the body binds none
   (FlambdaRegionsCheckSpec.all_code). *)
Definition code_checked (rs : region_state) : Prop :=
  distinct (map fst codes) = true
  ∧ ∀ (id : code_id) (c : code),
      lookup id (static_code (state_statics (rs_state rs))) = Some c ->
      lookup id codes = Some c ∧ check_code cert codes id c = true.

(* A closure that a frame binds is of code whose closures may be
   local. *)
Definition local_closures_fit (rs : region_state) : bool :=
  forallb
    (fun '(_, allocated) =>
       match allocated with
       | Static_closure code _ => closure_may_be_local cert code
       | Static_closure_of_set _ _ function_decls _ =>
         forallb (fun '(_, code) => closure_may_be_local cert code)
           function_decls
       | _ => true
       end)
    (live_locals (rs_frames rs)).

(* What each frame binds reaches symbols of the heap, of that frame or
   of a frame below it. The newest frame is first. *)
Fixpoint frames_deep (rs : region_state) (frames : list frame) : bool :=
  match frames with
  | [] => true
  | f :: below =>
    forallb
      (fun '(_, allocated) =>
         allocated_all (fun s => heap_symbol rs s || local_in (f :: below) s)
           allocated)
      (frame_locals f)
    && frames_deep rs below
  end.

(* What the static environment of the state binds reaches symbols of
   the heap. *)
Definition statics_deep (rs : region_state) : bool :=
  forallb (fun '(_, allocated) => allocated_all (heap_symbol rs) allocated)
    (static_values (state_statics (rs_state rs))).

(* The arguments that a partial application keeps against the
   parameters of its code: one that is not of the heap is at a
   parameter that may be given a local value. *)
Fixpoint values_fit (rs : region_state) (flags : list bool)
    (args : list value) : bool :=
  match args with
  | [] => true
  | v :: args =>
    match flags with
    | [] => false
    | flag :: flags =>
      (value_all (heap_symbol rs) v || flag) && values_fit rs flags args
    end
  end.

(* A partial application that may be called. It keeps fewer arguments
   than its code takes, and the code is not tupled: that is what a
   call makes. It reaches symbols of the heap only; or the certificate
   says that one may keep a local value, and what it keeps fits the
   code of its callee. One whose callee is a closure of code that the
   check was not given is never called. Nor is one whose callee is not
   a closure and cannot come to be one: a value that is no symbol, or
   a symbol that the machine drew, which is bound to what it was bound
   to or is gone. A callee that is a symbol of the term, bound to no
   closure, is refused: a [Let_static] could bind it later, and no
   call makes such a partial application. *)
Definition pap_fits (rs : region_state) (callee : value) (args : list value)
    (args_arity : arity) : bool :=
  match closure_of (live_statics rs) callee with
  | Some (id, _) =>
    match lookup id codes, lookup id (cert_code cert) with
    | Some c, Some entry =>
      negb (code_is_tupled c)
      && (length args_arity <? length (code_params_arity c))
      && ((value_all (heap_symbol rs) callee
           && forallb (value_all (heap_symbol rs)) args)
          || (cert_partial_local cert
              && values_fit rs (ce_params_local entry) args))
    | _, _ => true
    end
  | None =>
    match callee with
    | Value_symbol s => drawn rs s
    | _ => true
    end
  end.

(* Every partial application in a value passes [ok], however deep. *)
Fixpoint value_paps (ok : value -> list value -> arity -> bool) (v : value)
    : bool :=
  match v with
  | Value_const _ | Value_boxed_number _ | Value_location _
  | Value_symbol _ =>
    true
  | Value_closure _ value_slots | Value_closure_of_set _ _ value_slots =>
    forallb (fun '(_, v) => value_paps ok v) value_slots
  | Value_block _ fields => forallb (value_paps ok) fields
  | Value_immutable_array elements => forallb (value_paps ok) elements
  | Value_partial_application callee args args_arity =>
    ok callee args args_arity
    && value_paps ok callee
    && forallb (value_paps ok) args
  end.

Definition allocated_paps (ok : value -> list value -> arity -> bool)
    (allocated : statically_allocated_value) : bool :=
  match allocated with
  | Static_closure _ value_slots | Static_closure_of_set _ _ _ value_slots =>
    forallb (fun '(_, v) => value_paps ok v) value_slots
  | Static_block _ fields => forallb (value_paps ok) fields
  | Static_boxed_number contents => value_paps ok contents
  | Static_string _ | Static_imported _ => true
  | Static_immutable_array elements => forallb (value_paps ok) elements
  end.

(* The same of what a continuation holds: the variables of its
   closure, its extra arguments, the arguments that remain of an
   application with too many, and what the continuations that it holds
   hold. *)
Fixpoint cont_paps (ok : value -> list value -> arity -> bool)
    (c : cont_value) : bool :=
  match c with
  | Closure _ vars conts _ =>
    forallb (fun '(_, v) => value_paps ok v) vars
    && forallb (fun '(_, c) => cont_paps ok c) conts
  | Outer _ => true
  | Extra_args c extra => cont_paps ok c && forallb (value_paps ok) extra
  | Over_application remaining _ _ return_to exn_to =>
    forallb (value_paps ok) remaining
    && cont_paps ok return_to
    && cont_paps ok exn_to
  end.

(* Every partial application that the state holds may be called. *)
Definition paps_fit (rs : region_state) : bool :=
  let s := rs_state rs in
  let ok := pap_fits rs in
  forallb (fun '(_, v) => value_paps ok v) (state_vars s)
  && forallb (fun '(_, c) => cont_paps ok c) (state_conts s)
  && forallb (fun '(_, allocated) => allocated_paps ok allocated)
       (static_values (state_statics s))
  && forallb (fun '(_, allocated) => allocated_paps ok allocated)
       (live_locals (rs_frames rs)).

(* ======================================================================== *)
(* The property                                                             *)
(* ======================================================================== *)

Definition state_checked (rs : region_state) : Prop :=
  region_invariant rs = true
  ∧ frames_ordered rs = true
  ∧ code_checked rs
  ∧ local_closures_fit rs = true
  ∧ frames_deep rs (rs_frames rs) = true
  ∧ statics_deep rs = true
  ∧ paps_fit rs = true
  ∧ ∃ (cs : check_state) (ms : list mark),
      check_expr cert codes cs (state_expr (rs_state rs)) = true
      ∧ incl (code_of_expr (state_expr (rs_state rs))) codes
      ∧ running_checked rs cs ms
      ∧ conts_checked rs (running_user rs cs ms) (cs_conts cs)
          (state_conts (rs_state rs)).

End Checked.

(* ======================================================================== *)
(* What the other half of the proof needs                                   *)
(* ======================================================================== *)

(* The simples that a step evaluates: one at the place of a region has
   for its value a mark that is on the stack; the value of any other
   reaches only symbols that may be evaluated. *)
Fixpoint simples_live (rs : region_state) (simples : list simple)
    (places : list bool) : bool :=
  match simples with
  | [] => true
  | s :: simples =>
    (if hd false places then simple_open_mark rs s
     else
       match eval_simple (state_vars (rs_state rs)) s with
       | Some v => value_all (live_symbol rs) v
       | None => true
       end)
    && simples_live rs simples (tl places)
  end.

(* The continuations that the step of a state may apply: the one that
   an [Apply_cont] names; the return continuation of an [Apply], which
   is applied in the step of the call where the call makes a partial
   application. *)
Definition head_conts (rs : region_state) : list cont_value :=
  let s := rs_state rs in
  match state_expr s with
  | Apply ap => [resolve (ap_continuation ap) (state_conts s)]
  | Apply_cont ac => [resolve (ac_continuation ac) (state_conts s)]
  | _ => []
  end.

(* The values that applying a continuation gives on, or calls with:
   its extra arguments; the arguments that remain of an application
   with too many, and then, where the call of the value with them
   makes a partial application, those of the continuation that this
   is given to. FlambdaSemanticsSpec.apply_cont_value goes down the
   same way. *)
Fixpoint cont_spine_values (c : cont_value) : list value :=
  match c with
  | Closure _ _ _ _ | Outer _ => []
  | Extra_args c extra => extra ++ cont_spine_values c
  | Over_application remaining _ _ return_to _ =>
    remaining ++ cont_spine_values return_to
  end.

(* What a step reads may be read. Where [step] reads into a value (the
   callee inside a partial application, a field, a value slot, the
   tuple that tupled code is given), what it finds may be evaluated:
   - the simples of the head are as [simples_live] says;
   - what the static environment and the frames bind reaches only
     symbols that may be evaluated;
   - so do the values that the step gives on from a continuation that
     it applies: when a call returns into what is left of an
     application with too many arguments, the arguments that remain
     are those of a call, and the first is read where the code called
     is tupled. *)
Definition reads_live (rs : region_state) : Prop :=
  let s := rs_state rs in
  simples_live rs (evaluated_simples (state_expr s)) (region_places rs) = true
  ∧ forallb
      (fun '(_, allocated) => allocated_all (live_symbol rs) allocated)
      (static_values (live_statics rs))
    = true
  ∧ forallb (value_all (live_symbol rs))
      (flat_map cont_spine_values (head_conts rs))
    = true.

(* ======================================================================== *)
(* Statements                                                               *)
(* ======================================================================== *)

(* The state that a run of a checked term starts in has the
   property. *)
Definition checked_initial_statement : Prop :=
  ∀ (cert : certificate) (statics : static_env) (base : symbol) (e : expr),
    region_check cert statics e = true ->
    region_names_fresh base e statics = true ->
    state_checked cert (all_code statics e)
      (initial_region_state e statics base).

(* A step keeps it. *)
Definition region_step_keeps_checked_statement : Prop :=
  ∀ (cert : certificate) (codes : list (code_id * code))
    (rs rs' : region_state),
    state_checked cert codes rs ->
    region_step rs = Region_step rs' ->
    state_checked cert codes rs'.

(* A state that has it stops without a step only where [step] has
   none. *)
Definition checked_fault_statement : Prop :=
  ∀ (cert : certificate) (codes : list (code_id * code)) (rs : region_state)
    (fault : region_fault),
    state_checked cert codes rs ->
    region_step rs = Region_undefined fault ->
    fault = No_plain_step.

(* In a state that has it, what a step reads may be read. *)
Definition checked_reads_live_statement : Prop :=
  ∀ (cert : certificate) (codes : list (code_id * code)) (rs : region_state),
    state_checked cert codes rs -> reads_live rs.

(* The other half. Of a run in every state of which what a step reads
   may be read: where it stops because [step] has no step, the run of
   the term with its regions dropped goes wrong. Without the premise
   it is false: a partial application whose callee is gone may be
   evaluated, and calling it has no step, where the term with its
   regions dropped calls a closure; a value that is no mark, given at
   the place of a region, has no step for the kinds, where the term
   with its regions dropped has neither the parameter nor the
   argument. *)
Definition no_plain_step_erased_statement : Prop :=
  ∀ (e : expr) (statics : static_env) (base : symbol) (fuel : nat),
    region_names_fresh base e statics = true ->
    erasable e statics = true ->
    (∀ (fuel' : nat) (rs : region_state),
       region_run fuel' (initial_region_state e statics base)
       = Region_step rs ->
       reads_live rs) ->
    region_run fuel (initial_region_state e statics base)
    = Region_undefined No_plain_step ->
    behaves (erased_initial_state e statics) Goes_wrong.
