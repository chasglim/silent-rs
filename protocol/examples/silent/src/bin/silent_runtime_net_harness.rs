use std::env;
use std::net::TcpListener as StdTcpListener;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use silent_net::frame::MessageKind;
use silent_net::transport::tcp::TcpTransport;
use silent_net::transport::{Connection, Stream};
use silent_protocol_runtime::{
    FieldType, PartyId, RuntimeChannel, RuntimeConfig, RuntimeContext, RuntimeError, RuntimeMesh,
    RuntimeSession, SessionId, TaskId, Value, WireContext,
};
use tokio::runtime::Builder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = HarnessOptions::parse()?;
    if let Some(party) = opts.party {
        run_party(party, &opts)
    } else {
        run_orchestrator(&opts)
    }
}

#[derive(Clone, Debug)]
struct HarnessOptions {
    iterations: usize,
    mode: String,
    party: Option<u8>,
    listen: Option<String>,
    connect: Option<String>,
    no_header: bool,
}

impl HarnessOptions {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut opts = Self {
            iterations: 1,
            mode: "smoke".to_string(),
            party: None,
            listen: None,
            connect: None,
            no_header: false,
        };
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--iterations" | "-n" => {
                    opts.iterations = args
                        .next()
                        .ok_or("missing value for --iterations")?
                        .parse()?;
                }
                "--mode" | "--profile" => {
                    opts.mode = args.next().ok_or("missing value for --mode")?;
                    if opts.mode != "smoke" && opts.mode != "full" {
                        return Err("mode must be smoke or full".into());
                    }
                }
                "--party" => {
                    let party = args.next().ok_or("missing value for --party")?.parse()?;
                    if party > 1 {
                        return Err("party must be 0 or 1".into());
                    }
                    opts.party = Some(party);
                }
                "--listen" => {
                    opts.listen = Some(args.next().ok_or("missing value for --listen")?);
                }
                "--connect" => {
                    opts.connect = Some(args.next().ok_or("missing value for --connect")?);
                }
                "--no-header" => {
                    opts.no_header = true;
                }
                other => {
                    return Err(format!("unknown argument: {other}").into());
                }
            }
        }
        if opts.iterations == 0 {
            return Err("iterations must be positive".into());
        }
        Ok(opts)
    }
}

fn run_orchestrator(opts: &HarnessOptions) -> Result<(), Box<dyn std::error::Error>> {
    if !opts.no_header {
        println!("section,case,iterations,total_ms,avg_ms,extra");
    }
    let endpoint = reserve_loopback_endpoint()?;
    let exe = env::current_exe()?;
    let iterations = opts.iterations.to_string();

    let server = Command::new(&exe)
        .args([
            "--party",
            "0",
            "--listen",
            &endpoint,
            "--iterations",
            &iterations,
            "--mode",
            &opts.mode,
            "--no-header",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    thread::sleep(Duration::from_millis(50));

    let client_out = Command::new(&exe)
        .args([
            "--party",
            "1",
            "--connect",
            &endpoint,
            "--iterations",
            &iterations,
            "--mode",
            &opts.mode,
            "--no-header",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;
    let server_out = server.wait_with_output()?;

    if !client_out.status.success() {
        return Err(format!(
            "party1 failed: {}",
            String::from_utf8_lossy(&client_out.stderr)
        )
        .into());
    }
    if !server_out.status.success() {
        return Err(format!(
            "party0 failed: {}",
            String::from_utf8_lossy(&server_out.stderr)
        )
        .into());
    }

    print!("{}", String::from_utf8_lossy(&server_out.stdout));
    Ok(())
}

fn run_party(party: u8, opts: &HarnessOptions) -> Result<(), Box<dyn std::error::Error>> {
    let endpoint = match party {
        0 => opts.listen.as_deref().ok_or("party 0 requires --listen")?,
        1 => opts
            .connect
            .as_deref()
            .ok_or("party 1 requires --connect")?,
        _ => return Err("party must be 0 or 1".into()),
    };
    let runtime = Builder::new_current_thread().enable_all().build()?;
    let stream = runtime.block_on(open_party_stream(party, endpoint))?;
    let mut session = runtime_session_for_party(party, stream)?;
    let value = runtime_roundtrip_value()?;
    let peer = PartyId(1 - party as u16);
    let start = Instant::now();

    for _ in 0..opts.iterations {
        runtime.block_on(async {
            if party == 0 {
                let _route = session
                    .send_value(peer, TaskId(31), MessageKind::Data, &value)
                    .await?;
                let (_route, got) = session.recv_value(peer).await?;
                if got != value {
                    return Err(RuntimeError::InvalidValue("party0 ping-pong mismatch"));
                }
            } else {
                let (_route, got) = session.recv_value(peer).await?;
                if got != value {
                    return Err(RuntimeError::InvalidValue("party1 ping-pong mismatch"));
                }
                let _route = session
                    .send_value(peer, TaskId(31), MessageKind::Data, &got)
                    .await?;
            }
            Ok::<_, RuntimeError>(())
        })?;
    }

    if party == 0 {
        let elapsed = start.elapsed();
        let stats = session.stats();
        print_row(
            "runtime",
            "silent_net_tcp_multiprocess_pingpong_u64x8",
            opts.iterations,
            elapsed,
            &format!(
                "profile={},transport=tcp_multiprocess,party=0,peer=1,endpoint={},frames_sent={},frames_received={},bytes_sent={},bytes_received={}",
                opts.mode,
                endpoint,
                stats.frames_sent,
                stats.frames_received,
                stats.bytes_sent,
                stats.bytes_received
            ),
        );
    }
    Ok(())
}

async fn open_party_stream(party: u8, endpoint: &str) -> Result<Box<dyn Stream>, RuntimeError> {
    if party == 0 {
        let listener = TcpTransport::new()
            .bind(endpoint)
            .await
            .map_err(|err| RuntimeError::Network(err.to_string()))?;
        let conn = listener
            .accept()
            .await
            .map_err(|err| RuntimeError::Network(err.to_string()))?;
        conn.open_stream()
            .await
            .map_err(|err| RuntimeError::Network(err.to_string()))
    } else {
        connect_with_retry(endpoint).await
    }
}

async fn connect_with_retry(endpoint: &str) -> Result<Box<dyn Stream>, RuntimeError> {
    let mut last_error = None;
    for _ in 0..100 {
        match TcpTransport::new().connect(endpoint).await {
            Ok(conn) => {
                return conn
                    .open_stream()
                    .await
                    .map_err(|err| RuntimeError::Network(err.to_string()));
            }
            Err(err) => {
                last_error = Some(err);
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
    }
    Err(RuntimeError::Network(format!(
        "failed to connect to {endpoint}: {}",
        last_error
            .map(|err| err.to_string())
            .unwrap_or_else(|| "unknown error".to_string())
    )))
}

fn runtime_session_for_party(
    party: u8,
    stream: Box<dyn Stream>,
) -> Result<RuntimeSession, RuntimeError> {
    let local = PartyId(party as u16);
    let peer = PartyId(1 - party as u16);
    let wire = WireContext::default();
    let cfg = RuntimeConfig::two_party_inference();
    let ctx = RuntimeContext::new(cfg, local, SessionId(3030), wire.clone())?;
    let mut mesh = RuntimeMesh::new(local);
    mesh.insert_peer(peer, RuntimeChannel::new(stream, wire))?;
    RuntimeSession::with_mesh(ctx, mesh)
}

fn runtime_roundtrip_value() -> Result<Value, RuntimeError> {
    Value::public_u64(
        vec![3, 5, 8, 13, 21, 34, 55, 89],
        FieldType::Ring64,
        vec![8],
    )
}

fn reserve_loopback_endpoint() -> Result<String, Box<dyn std::error::Error>> {
    let listener = StdTcpListener::bind("127.0.0.1:0")?;
    let endpoint = listener.local_addr()?.to_string();
    drop(listener);
    Ok(endpoint)
}

fn print_row(section: &str, case: &str, iterations: usize, total: Duration, extra: &str) {
    let total_ms = total.as_secs_f64() * 1000.0;
    let avg_ms = total_ms / iterations as f64;
    println!("{section},{case},{iterations},{total_ms:.3},{avg_ms:.3},{extra}");
}
