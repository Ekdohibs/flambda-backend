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

(* The demotion of an exception handler: a handler that no call can
   raise to is made an ordinary continuation, and the trap actions that
   name it are removed.

   This file is to be reviewed: [called_with_exn], [binds_continuation],
   [demotable], [strip_traps], [clear_exn_handler] and
   [demote_handlers], against what Simplify does
   (simplify/simplify_let_cont_expr.ml, where a handler whose uses do
   not escape is recorded by [DA.demote_exn_handler] and rebuilt as an
   ordinary one; simplify/simplify_common.ml: [apply_cont_use_kind],
   which says which uses escape, and [clear_demoted_trap_action], which
   removes the trap actions); and the statements.

   Where Simplify sees every raise that reaches a handler, the handler
   need not be installed: a raise to it can be a jump. Simplify does
   this for every [try] of the source whose body it can read to the
   end. The exception continuation of a call is a use that escapes: the
   callee finds the handler on the stack. So the handler of a [try]
   with a call in its body, outside any inner [try], stays.

   [demote_handlers] is that rewrite, done wherever it applies. At a
   non-recursive [Let_cont] of [k] whose handler says it is an
   exception handler, if no call in the body has [k] as its exception
   continuation ([demotable]), the handler loses its mark and every
   trap action of the body that names [k] is removed: the [Push] that
   installed it, the [Pop] on the way out of the [try], and the [Pop]
   of each raise to it, which becomes an ordinary application of [k].
   It goes into handlers and into code.

   The rewrite, like the checks of exception handlers, goes by names
   and does not follow rebinding. So it leaves alone a handler whose
   name is bound again in the body, or is that of a continuation in
   whose scope the [Let_cont] stands: the function takes the names in
   scope ([scope]), which for a unit are its return and exception
   continuations, and for code those of the code. The compiler's terms
   bind each name once, and lose nothing by this.

   What it does not look at, where Simplify does:
   - the kind of a raise. Simplify takes a [Pop] of a raise that keeps a
     backtrace ([Regular] or [Reraise]) for a use that escapes when
     the unit is compiled with debugging information, and so keeps the
     handler; the machines observe no backtrace, and the rewrite does
     not look (discrepancy C5-4). On a unit compiled so the rewrite
     demotes where Simplify did not, which can leave the two terms
     different in structure. Nothing is claimed here of backtraces, or
     of the native code of a handler that is demoted;
   - the argument of a raise. Where nothing reads the exception,
     Simplify replaces it ([patch_unused_exn_bucket]); that is another
     rewrite.

   The statements.
   - [demote_handlers_equivalent_statement]: the term and the result
     behave alike under all environments. This is about the machine of
     FlambdaSemanticsSpec.v, which executes no trap action and does not
     read the mark of a handler: on that machine any change of trap
     actions and of marks is an equivalence, and the condition
     ([demotable]) plays no part.
   - The condition is what the next three need. A unit that passes the
     two checks of exception handlers
     (FlambdaWellFormedSpec.unit_exn_handlers_well_formed and
     traps_well_formed) still passes them after the rewrite
     ([demote_handlers_keeps_exn_handlers_statement],
     [demote_handlers_keeps_traps_statement]); and so, on the machine
     with a stack of handlers (FlambdaTrapSpec.v), a unit that follows
     the discipline of trap actions behaves after the rewrite as before
     ([demote_handlers_stack_statement]). Without the condition the
     last is false: a handler demoted under a call no longer catches
     what the callee raises ([demotion_under_call_statement]).
   - The rewrite keeps the binders of a term
     ([demote_handlers_binders_statement]), and commutes with renaming
     ([demote_handlers_equivariant_statement]).
   That it keeps scope, in the sense that the validator asks of a
   rewrite it applies, is stated in FlambdaDemoteScopeSpec.v. The
   statement about the stack is from the empty static environment, as
   FlambdaTrapSpec.stack_agrees_statement is;
   [demote_handlers_stack_from_statement] is the same from a static
   environment that holds no code, as
   FlambdaTrapSpec.stack_agrees_from_statement is.

   Why the condition is enough, in terms of the check of trap actions.
   Under the discipline a call names the handler installed last. So
   "no call in the scope of [k] names [k]" says that no call is made
   while [k] is on top of the stack. Remove [k] from the stack
   everywhere: a call still has its own handler on top; a [Pop] of
   another handler still finds it on top; a raise to [k] found [k] on
   top and what was below it is the stack at the [Push], which is the
   stack that the jump now arrives with.

   The model has no exception that is not raised by a raise or a call
   of the term. The compiler has some: stdlib/sys.mli documents that an
   asynchronous [Break] or [Stack_overflow] (from a signal handler, a
   finaliser or the collector) is raised from the call site of
   [Sys.with_async_exns], and without one goes to the toplevel handler
   of uncaught exceptions. A [try] around that call may catch it; but
   then its body has a call whose exception continuation is its
   handler, and the rewrite leaves it alone. So, of those two
   exceptions, the handler of a [try] whose body has no call is
   reached by the raises of the body alone, as the rewrite takes it.
   Any other event of the runtime is outside the model.

   Two examples: the [try] that the front end makes around a raise,
   which the rewrite makes into what Simplify leaves
   ([demote_handlers_try_raise_statement]); and the one above, of a
   [try] around a call.

   The proofs are in FlambdaDemoteProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaBindersSpec FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaWellFormedSpec FlambdaTrapSpec.
Import ListNotations.

(* ======================================================================== *)
(* The condition                                                            *)
(* ======================================================================== *)

(* Whether some call of the expression has [k] as its exception
   continuation. Code is closed, so bindings of code have none. *)
Fixpoint called_with_exn (k : continuation) (e : expr) : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => called_with_exn k body
  | Let_cont_non_recursive _ h body =>
    handler_called_with_exn k h || called_with_exn k body
  | Let_cont_recursive handlers body =>
    existsb (fun '(_, h) => handler_called_with_exn k h) handlers
    || called_with_exn k body
  | Apply ap => Pos.eqb k (ap_exn_continuation ap)
  | Apply_cont _ | Switch _ _ | Invalid => false
  end
with handler_called_with_exn (k : continuation) (h : continuation_handler)
    : bool :=
  match h with
  | Handler _ _ handler => called_with_exn k handler
  end.

(* Whether the expression binds the continuation [k], at any depth
   outside code. *)
Fixpoint binds_continuation (k : continuation) (e : expr) : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => binds_continuation k body
  | Let_cont_non_recursive k' h body =>
    Pos.eqb k k' || handler_binds_continuation k h
    || binds_continuation k body
  | Let_cont_recursive handlers body =>
    existsb
      (fun '(k', h) => Pos.eqb k k' || handler_binds_continuation k h)
      handlers
    || binds_continuation k body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => false
  end
with handler_binds_continuation (k : continuation)
    (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => binds_continuation k handler
  end.

(* The handler [h] of [k], around [body], where the continuations
   [scope] are in scope, may be demoted: it says it is an exception
   handler, no call of [body] has [k] as its exception continuation,
   and [k] is no name of [scope] and is not bound again in [body]. *)
Definition demotable (scope : list continuation) (k : continuation)
    (h : continuation_handler) (body : expr) : bool :=
  handler_is_exn_handler h
  && negb (called_with_exn k body)
  && negb (among k scope)
  && negb (binds_continuation k body).

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* An [Apply_cont] without its trap action, if that names [k]. *)
Definition strip_trap (k : continuation) (ac : apply_cont) : apply_cont :=
  if traps k ac then Apply_cont_expr (ac_continuation ac) (ac_args ac) None
  else ac.

(* An expression without the trap actions that name [k]. It does not go
   into code, which is closed. *)
Fixpoint strip_traps (k : continuation) (e : expr) : expr :=
  match e with
  | Let x kind defining_expr body =>
    Let x kind defining_expr (strip_traps k body)
  | Let_static bindings body => Let_static bindings (strip_traps k body)
  | Let_cont_non_recursive k' h body =>
    Let_cont_non_recursive k' (strip_traps_handler k h) (strip_traps k body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun kh => (fst kh, strip_traps_handler k (snd kh))) handlers)
      (strip_traps k body)
  | Apply ap => Apply ap
  | Apply_cont ac => Apply_cont (strip_trap k ac)
  | Switch scrutinee arms =>
    Switch scrutinee (map (fun arm => (fst arm, strip_trap k (snd arm))) arms)
  | Invalid => Invalid
  end
with strip_traps_handler (k : continuation) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (strip_traps k handler)
  end.

(* A handler that does not say it is an exception handler. *)
Definition clear_exn_handler (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params (make_handler_metadata (hm_params_kinds metadata)) handler
  end.

(* The expression with every handler demoted that may be, in its
   handlers and in its code too, where the continuations [scope] are in
   scope. The test is on the expression as it is given: the rewrite
   does not change what the test reads. A handler of a non-recursive
   [Let_cont] is not in the scope of its continuation, the handlers of
   a recursive one are in the scope of them all, and the body of code
   is in the scope of its return and exception continuations alone. *)
Fixpoint demote_handlers (scope : list continuation) (e : expr) : expr :=
  match e with
  | Let x kind defining_expr body =>
    Let x kind defining_expr (demote_handlers scope body)
  | Let_static bindings body =>
    Let_static (map demote_handlers_binding bindings)
      (demote_handlers scope body)
  | Let_cont_non_recursive k h body =>
    if demotable scope k h body
    then
      Let_cont_non_recursive k
        (clear_exn_handler (demote_handlers_handler scope h))
        (strip_traps k (demote_handlers (k :: scope) body))
    else
      Let_cont_non_recursive k (demote_handlers_handler scope h)
        (demote_handlers (k :: scope) body)
  | Let_cont_recursive handlers body =>
    let inner := map fst handlers ++ scope in
    Let_cont_recursive
      (map (fun kh => (fst kh, demote_handlers_handler inner (snd kh)))
         handlers)
      (demote_handlers inner body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with demote_handlers_handler (scope : list continuation)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (demote_handlers scope handler)
  end
with demote_handlers_binding (binding : static_binding) : static_binding :=
  match binding with
  | Bind_code id c => Bind_code id (demote_handlers_code c)
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end
with demote_handlers_code (c : code) : code :=
  match c with
  | Code params my_closure return_continuation exn_continuation metadata
      body =>
    Code params my_closure return_continuation exn_continuation metadata
      (demote_handlers [return_continuation; exn_continuation] body)
  end.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* On the machine that goes by names, the result behaves as the term,
   under all environments. *)
Definition demote_handlers_equivalent_statement : Prop :=
  ∀ (scope : list continuation) (e : expr),
    expr_equivalent e (demote_handlers scope e).

(* The rewrite keeps the binders. *)
Definition demote_handlers_binders_statement : Prop :=
  ∀ (scope : list continuation) (e : expr),
    binders_distinct (demote_handlers scope e) = binders_distinct e.

(* The rewrite commutes with renaming, the names in scope being renamed
   with the term: each of its tests compares two names, and a renaming
   is a permutation of the names. *)
Definition demote_handlers_equivariant_statement : Prop :=
  ∀ (r : renaming) (scope : list continuation) (e : expr),
    demote_handlers (map (rename_continuation r) scope) (rename_expr r e)
    = rename_expr r (demote_handlers scope e).

(* A unit whose exception handlers are apart from its other
   continuations is still such a unit. Its exception continuation is to
   be among the names in scope. *)
Definition demote_handlers_keeps_exn_handlers_statement : Prop :=
  ∀ (unit_exn : continuation) (scope : list continuation) (e : expr),
    In unit_exn scope ->
    unit_exn_handlers_well_formed unit_exn e = true ->
    unit_exn_handlers_well_formed unit_exn (demote_handlers scope e) = true.

(* A unit whose binders are distinct and that follows the discipline of
   trap actions still follows it. *)
Definition demote_handlers_keeps_traps_statement : Prop :=
  ∀ (unit_exn : continuation) (scope : list continuation) (e : expr),
    binders_distinct e = true ->
    traps_well_formed unit_exn e = true ->
    traps_well_formed unit_exn (demote_handlers scope e) = true.

(* On the machine with a stack of handlers, a unit whose binders are
   distinct and that follows the discipline of trap actions behaves
   after the rewrite as before. *)
Definition demote_handlers_stack_statement : Prop :=
  ∀ (unit_exn : continuation) (scope : list continuation) (e : expr)
    (b : behaviour),
    binders_distinct e = true ->
    traps_well_formed unit_exn e = true ->
    stack_behaves
      (State (demote_handlers scope e) [] [] empty_static_env empty_heap
         no_reads)
      (initial_stack unit_exn) b
    <-> stack_behaves (State e [] [] empty_static_env empty_heap no_reads)
          (initial_stack unit_exn) b.

(* The same of a unit run from a static environment that holds no
   code, as that of the symbols that a unit imports is
   (FlambdaTrapSpec.stack_agrees_from_statement). *)
Definition demote_handlers_stack_from_statement : Prop :=
  ∀ (statics : static_env) (unit_exn : continuation)
    (scope : list continuation) (e : expr) (b : behaviour),
    static_code statics = [] ->
    binders_distinct e = true ->
    traps_well_formed unit_exn e = true ->
    stack_behaves
      (State (demote_handlers scope e) [] [] statics empty_heap no_reads)
      (initial_stack unit_exn) b
    <-> stack_behaves (State e [] [] statics empty_heap no_reads)
          (initial_stack unit_exn) b.

(* ======================================================================== *)
(* Two examples                                                             *)
(* ======================================================================== *)

(* [try raise with _ -> 2], as the front end makes it when [installed],
   and as Simplify leaves it when not: the handler is installed on the
   way into the body, and the body raises to it. *)
Definition try_raise (installed : bool) : expr :=
  let leave : continuation := 2%positive in
  let handler : continuation := 3%positive in
  let body : continuation := 4%positive in
  let x : variable := 1%positive in
  let action (a : trap_action) : option trap_action :=
    if installed then Some a else None
  in
  Let_cont_non_recursive handler
    (Handler [x]
       (if installed then make_exn_handler_metadata [Value]
        else make_handler_metadata [Value])
       (Apply_cont
          (Apply_cont_expr leave [Const (Tagged_immediate 2%Z)] None)))
    (Let_cont_non_recursive body
       (Handler [] (make_handler_metadata [])
          (Apply_cont
             (Apply_cont_expr handler [Const (Tagged_immediate 0%Z)]
                (action (Pop handler (Some Regular))))))
       (Apply_cont (Apply_cont_expr body [] (action (Push handler))))).

(* The rewrite makes the second of the first. *)
Definition demote_handlers_try_raise_statement : Prop :=
  let unit_exn : continuation := 1%positive in
  let leave : continuation := 2%positive in
  demote_handlers [leave; unit_exn] (try_raise true) = try_raise false.

(* [try f () with _ -> 2], where [f] raises: the call is in the body of
   the [try], and its exception continuation is the handler. With
   [installed] it is as the front end makes it. Without, the handler
   has been demoted as [demote_handlers] would not: it has lost its
   mark and the trap actions that name it are gone. *)
Definition try_around_call (installed : bool) : expr :=
  let leave : continuation := 2%positive in
  let handler : continuation := 3%positive in
  let body : continuation := 4%positive in
  let after : continuation := 5%positive in
  let ret : continuation := 6%positive in
  let exn : continuation := 7%positive in
  let f : variable := 1%positive in
  let x : variable := 2%positive in
  let id : code_id := 1%positive in
  let action (a : trap_action) : option trap_action :=
    if installed then Some a else None
  in
  Let_static
    [Bind_code id
       (Code [] f ret exn (make_code_metadata [] [Singleton Value])
          (Apply_cont
             (Apply_cont_expr exn [Const (Tagged_immediate 0%Z)]
                (Some (Pop exn (Some No_trace))))))]
    (Let f Value (Set_of_closures id [])
       (Let_cont_non_recursive handler
          (Handler [x]
             (if installed then make_exn_handler_metadata [Value]
              else make_handler_metadata [Value])
             (Apply_cont
                (Apply_cont_expr leave [Const (Tagged_immediate 2%Z)] None)))
          (Let_cont_non_recursive body
             (Handler [] (make_handler_metadata [])
                (Let_cont_non_recursive after
                   (Handler [x] (make_handler_metadata [Value])
                      (Apply_cont
                         (Apply_cont_expr leave [Var x]
                            (action (Pop handler None)))))
                   (Apply
                      (Apply_expr (Var f) (Direct id) [] after handler [] []
                         [Singleton Value]))))
             (Apply_cont (Apply_cont_expr body [] (action (Push handler))))))).

(* The condition is needed. The rewrite leaves the [try] around a call
   alone. Were its handler demoted, the machine with a stack of
   handlers would no longer bring the exception of the callee to it:
   the unit leaves with 2 as it is, and by its exception continuation,
   with the exception, once the handler is demoted. *)
Definition demotion_under_call_statement : Prop :=
  let unit_exn : continuation := 1%positive in
  let leave : continuation := 2%positive in
  let before := try_around_call true in
  let after := try_around_call false in
  binders_distinct before = true
  ∧ traps_well_formed unit_exn before = true
  ∧ demote_handlers [leave; unit_exn] before = before
  ∧ stack_behaves (State before [] [] empty_static_env empty_heap no_reads)
      (initial_stack unit_exn)
      (Exits leave [Value_const (Tagged_immediate 2%Z)] empty_heap)
  ∧ stack_behaves (State after [] [] empty_static_env empty_heap no_reads)
      (initial_stack unit_exn)
      (Exits unit_exn [Value_const (Tagged_immediate 0%Z)] empty_heap).
